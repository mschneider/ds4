/* Next-layer expert prediction for SSD streaming (fork), Metal side.
 * ds4_qwen4_expert_predict.inc (ds4.c) encodes layer l+1's router on layer
 * l's MoE input and registers the predicted ids here; after layer l's
 * experts are submitted, the streamed MoE starts loading the predicted
 * experts that are not cached with the asynchronous early-load path. The
 * next layer's streamed MoE joins and installs them after its GPU sync
 * (ds4_gpu_stream_expert_pending_load_clear).
 *
 * Layer l's queued GPU work still reads its own experts, so they are marked
 * in flight before the loads choose victims; the Qwen stream path otherwise
 * relies on its per-layer sync instead of in-flight marks.
 *
 * The guess pays only while the cache is cold. With a warm cache the
 * missing experts are mostly the ones the guess does not foresee (on an M4 Pro
 * with a 13 GiB cache after a 4K prompt, 11.5% of the loaded guesses were
 * used and decode slowed down), so the loads pause for a while whenever fewer
 * than DS4_QWEN4_STREAM_PREDICT_MIN_USE (default 0.25) of the last
 * QWEN4_PREDICT_WINDOW loaded guesses were used.
 *
 * Included by ds4_metal.m before ds4_gpu_qwen4_moe_stream_tensor.
 * DS4_QWEN4_STREAM_PREDICT_PROFILE=1 prints the prediction's accuracy. */

#define QWEN4_PREDICT_WINDOW 128u
#define QWEN4_PREDICT_PAUSE_STEPS (128u * 47u)  /* about 128 decode tokens */

static ds4_gpu_stream_expert_table g_qwen4_predict_table;
static const ds4_gpu_tensor *g_qwen4_predict_ids;
static uint32_t g_qwen4_predict_n;

/* The last prediction, to score against the layer's real routing. */
static uint32_t g_qwen4_predict_scored_layer = UINT32_MAX;
static int32_t g_qwen4_predict_guess[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
static uint32_t g_qwen4_predict_n_guess;
static int32_t g_qwen4_predict_loaded[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
static uint32_t g_qwen4_predict_loaded_rank[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
static uint32_t g_qwen4_predict_n_loaded;
static struct {
    uint64_t layers, guessed, guessed_right, loaded, loaded_right, failed, pauses, paused_steps;
    uint64_t rank_loaded[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint64_t rank_right[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
} g_qwen4_predict_stats;

static uint32_t g_qwen4_predict_pause;
static uint32_t g_qwen4_predict_window_loaded, g_qwen4_predict_window_used;

static double qwen4_stream_predict_min_use(void) {
    static double v = -1.0;
    if (v < 0.0) {
        const char *env = getenv("DS4_QWEN4_STREAM_PREDICT_MIN_USE");
        v = env && env[0] ? strtod(env, NULL) : 0.25;
        if (!(v >= 0.0)) v = 0.25;
    }
    return v;
}

static int qwen4_stream_predict_profile(void) {
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("DS4_QWEN4_STREAM_PREDICT_PROFILE") != NULL;
    return enabled;
}

static void qwen4_stream_predict_report(void) {
    const typeof(g_qwen4_predict_stats) *s = &g_qwen4_predict_stats;
    fprintf(stderr,
            "ds4: Qwen expert prediction: layers=%llu guessed=%llu right=%.1f%% "
            "loaded=%llu used=%.1f%% failed=%llu pauses=%llu paused_steps=%llu\n",
            (unsigned long long)s->layers, (unsigned long long)s->guessed,
            s->guessed ? 100.0 * (double)s->guessed_right / (double)s->guessed : 0.0,
            (unsigned long long)s->loaded,
            s->loaded ? 100.0 * (double)s->loaded_right / (double)s->loaded : 0.0,
            (unsigned long long)s->failed, (unsigned long long)s->pauses,
            (unsigned long long)s->paused_steps);
    for (uint32_t r = 0; r < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED; r++) {
        if (!s->rank_loaded[r]) continue;
        fprintf(stderr, "ds4:   guess rank %u: loaded=%llu used=%.1f%%\n", r,
                (unsigned long long)s->rank_loaded[r],
                100.0 * (double)s->rank_right[r] / (double)s->rank_loaded[r]);
    }
}

int ds4_gpu_qwen4_stream_predict_set(const ds4_gpu_stream_expert_table *next,
                                     const ds4_gpu_tensor *ids, uint32_t n_ids) {
    if (!next || !ids || n_ids == 0 || n_ids > DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED) return 0;
    g_qwen4_predict_table = *next;
    g_qwen4_predict_ids = ids;
    g_qwen4_predict_n = n_ids;
    return 1;
}

static int qwen4_predict_contains(const int32_t *ids, uint32_t n, int32_t id) {
    for (uint32_t i = 0; i < n; i++) if (ids[i] == id) return 1;
    return 0;
}

/* Called at the end of a streamed MoE for layer `layer` with its routing,
 * while its command batch is still open. */
static void qwen4_stream_predict_after_layer(int ok, uint32_t layer,
                                             const int32_t *ids, uint32_t n_ids) {
    const int profile = qwen4_stream_predict_profile();
    if (ok && layer == g_qwen4_predict_scored_layer) {
        g_qwen4_predict_stats.layers++;
        for (uint32_t i = 0; profile && i < g_qwen4_predict_n_guess; i++) {
            g_qwen4_predict_stats.guessed++;
            g_qwen4_predict_stats.guessed_right += qwen4_predict_contains(ids, n_ids, g_qwen4_predict_guess[i]);
        }
        for (uint32_t i = 0; i < g_qwen4_predict_n_loaded; i++) {
            const int right = qwen4_predict_contains(ids, n_ids, g_qwen4_predict_loaded[i]);
            const uint32_t r = g_qwen4_predict_loaded_rank[i];
            g_qwen4_predict_stats.loaded++;
            g_qwen4_predict_stats.loaded_right += right;
            g_qwen4_predict_stats.rank_loaded[r]++;
            g_qwen4_predict_stats.rank_right[r] += right;
            g_qwen4_predict_window_loaded++;
            g_qwen4_predict_window_used += right;
        }
        if (g_qwen4_predict_window_loaded >= QWEN4_PREDICT_WINDOW) {
            if (g_qwen4_predict_window_used <
                qwen4_stream_predict_min_use() * g_qwen4_predict_window_loaded) {
                g_qwen4_predict_pause = QWEN4_PREDICT_PAUSE_STEPS;
                g_qwen4_predict_stats.pauses++;
            }
            g_qwen4_predict_window_loaded = 0;
            g_qwen4_predict_window_used = 0;
        }
    }
    g_qwen4_predict_scored_layer = UINT32_MAX;

    const ds4_gpu_tensor *pred = g_qwen4_predict_ids;
    const uint32_t n_pred = g_qwen4_predict_n;
    const ds4_gpu_stream_expert_table next = g_qwen4_predict_table;
    g_qwen4_predict_ids = NULL;
    if (!ok || !pred || next.layer != layer + 1u) return;
    if (g_qwen4_predict_pause) {
        g_qwen4_predict_pause--;
        g_qwen4_predict_stats.paused_steps++;
        return;
    }

    /* The prediction ran before this layer's sync, so it is complete. */
    int32_t guess[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    if (!ds4_gpu_tensor_read(pred, 0, guess, (uint64_t)n_pred * sizeof(*guess))) return;
    for (uint32_t i = 0; i < n_pred; i++) {
        if (guess[i] < 0 || (uint32_t)guess[i] >= next.n_total_expert) return;
    }
    for (uint32_t i = 0; i < n_ids; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) continue;
        ds4_gpu_stream_expert_cache_entry *e = &g_stream_expert_cache[layer][ids[i]];
        if (e->valid && !ds4_gpu_stream_expert_cache_mark_inflight(e)) return;
    }

    static int warned;
    if (!ds4_gpu_stream_expert_cache_begin_selected_load(&next, guess, n_pred)) {
        g_qwen4_predict_stats.failed++;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "ds4: warning: Qwen expert prediction load failed; continuing without it\n");
        }
        return;
    }
    static int registered;
    if (profile && !registered) {
        registered = 1;
        atexit(qwen4_stream_predict_report);
    }
    g_qwen4_predict_scored_layer = next.layer;
    memcpy(g_qwen4_predict_guess, guess, (size_t)n_pred * sizeof(*guess));
    g_qwen4_predict_n_guess = n_pred;
    const ds4_gpu_stream_expert_pending_load *p = &g_stream_expert_pending_load;
    g_qwen4_predict_n_loaded = 0;
    for (uint32_t i = 0; p->active && p->layer == next.layer && i < p->n_loads; i++) {
        g_qwen4_predict_loaded_rank[g_qwen4_predict_n_loaded] = p->load_slots[i];
        g_qwen4_predict_loaded[g_qwen4_predict_n_loaded++] = p->selected_ids[p->load_slots[i]];
    }
}
