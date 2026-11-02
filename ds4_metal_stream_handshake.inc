/* Event handshake for streamed Qwen decode layers (fork).
 *
 * The ordinary streamed MoE drains the GPU at every layer so the CPU can
 * read the router's expert ids, load the missing experts and bind them.
 * With DS4_QWEN4_STREAM_HANDSHAKE=1 a layer of one to three tokens (decode
 * and MTP verification) is instead encoded
 * without knowing its experts:
 *
 *   ... router(l) | signal(v) | wait(v + 1) | experts(l) via the layer's
 *   expert address table | reduce ...
 *
 * and committed at once. The CPU keeps encoding the next layer and only then
 * services layer l: it waits for v, reads the ids, loads what is missing
 * into the expert cache, writes their addresses into the table and sets
 * v + 1, which releases the GPU. The GPU thus runs ahead by a layer instead
 * of idling through every drain, wake-up and encode.
 *
 * Safety:
 * - The expert kernels reach the cache only through the address table and
 *   the slab buffers, which are all made resident at encode time; the
 *   handshake first allocates every slab of the cache budget.
 * - When the CPU sees v, the GPU has finished everything before router(l),
 *   so experts of earlier layers can be evicted; layer l's own selection is
 *   protected by the load. Handshake layers do not mark entries in flight,
 *   and the first handshake layer after an ordinary one drains once, so no
 *   load ever waits for in-flight GPU work (which would wait on the GPU
 *   that waits on the CPU).
 * - Every command-buffer wait first services pending layers
 *   (ds4_gpu_wait_command_buffer), so nothing can wait on a blocked GPU.
 * - A failed load still releases the GPU, with the failed experts pointed at
 *   valid memory, and fails the next wait; the token is not used.
 *
 * Included by ds4_metal.m before ds4_gpu_qwen4_moe_stream_tensor. */

#define QWEN4_HS_MAX_TOKENS 3u
#define QWEN4_HS_MAX_IDS (QWEN4_HS_MAX_TOKENS * DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED)

typedef struct {
    ds4_gpu_stream_expert_table table;
    uint64_t value;
    const volatile int32_t *ids;
    uint32_t n_ids;
} qwen4_handshake_layer;

static id<MTLSharedEvent> g_qwen4_hs_event;
static uint64_t g_qwen4_hs_value;
static qwen4_handshake_layer g_qwen4_hs_pending[2];
static uint32_t g_qwen4_hs_n_pending;
static int g_qwen4_hs_servicing;
static int g_qwen4_hs_failed;
static int g_qwen4_hs_dirty = 1;  /* an ordinary layer ran since the last drain */
/* DS4_QWEN4_STREAM_HANDSHAKE_PROFILE=1 prints these at exit. */
static struct {
    uint64_t layers, ordinary, not_full, drains;
    double wait_ms, service_ms;
} g_qwen4_hs_stats;

static void qwen4_handshake_report(void) {
    const typeof(g_qwen4_hs_stats) *s = &g_qwen4_hs_stats;
    fprintf(stderr, "ds4: Qwen decode handshake: layers=%llu ordinary=%llu (cache not full %llu) "
            "drains=%llu CPU wait for GPU %.3f ms/layer, service %.3f ms/layer\n",
            (unsigned long long)s->layers, (unsigned long long)s->ordinary,
            (unsigned long long)s->not_full, (unsigned long long)s->drains,
            s->layers ? s->wait_ms / (double)s->layers : 0.0,
            s->layers ? s->service_ms / (double)s->layers : 0.0);
}

static int qwen4_stream_handshake_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("DS4_QWEN4_STREAM_HANDSHAKE");
        enabled = env && strcmp(env, "1") == 0;
        if (enabled && getenv("DS4_QWEN4_STREAM_HANDSHAKE_PROFILE")) atexit(qwen4_handshake_report);
    }
    return enabled;
}

static void qwen4_handshake_fail(const char *why) {
    static int warned;
    if (!warned) {
        warned = 1;
        fprintf(stderr, "ds4: Qwen streamed decode handshake failed: %s\n", why);
    }
    g_qwen4_hs_failed = 1;
}

static void qwen4_handshake_service(const qwen4_handshake_layer *h) {
    const ds4_gpu_stream_expert_table *t = &h->table;
    const double t0 = ds4_gpu_now_ms();
    uint32_t spins = 0;
    while (g_qwen4_hs_event.signaledValue < h->value) {
        if (++spins > (1u << 14)) {
            spins = 0;
            sched_yield();
            if (ds4_gpu_now_ms() - t0 > 60000.0) {
                qwen4_handshake_fail("the GPU did not reach the layer's router");
                break;
            }
        }
    }

    const double t1 = ds4_gpu_now_ms();
    g_qwen4_hs_stats.wait_ms += t1 - t0;
    int32_t ids[QWEN4_HS_MAX_IDS];
    int32_t unique[QWEN4_HS_MAX_IDS];
    uint32_t n_unique = 0;
    int ok = !g_qwen4_hs_failed;
    uint32_t frequency[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT] = {0};
    for (uint32_t i = 0; ok && i < h->n_ids; i++) {
        ids[i] = h->ids[i];
        if (ids[i] < 0 || (uint32_t)ids[i] >= t->n_total_expert) ok = 0;
        else if (frequency[ids[i]]++ == 0) unique[n_unique++] = ids[i];
    }
    if (ok) {
        ds4_gpu_stream_expert_cache_note_token(t->layer);
        ds4_gpu_stream_expert_cache_note_frequency_hotness(t->layer, frequency, t->n_total_expert);
    }
    /* One load protects only its own selection, so a multi-token step loads
     * its experts in chunks and then checks that all of them are present,
     * reloading any that a later chunk evicted. */
    ds4_gpu_stream_expert_cache_entry *entry[QWEN4_HS_MAX_IDS];
    for (uint32_t attempt = 0; ok; attempt++) {
        int32_t missing[QWEN4_HS_MAX_IDS];
        uint32_t n_missing = 0;
        for (uint32_t i = 0; i < n_unique; i++) {
            const uint64_t e = (uint64_t)(uint32_t)unique[i];
            entry[i] = ds4_gpu_stream_expert_cache_peek(
                t->model_map, t->model_size, t->layer, (uint32_t)e, t->n_total_expert,
                n_unique < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED ?
                    n_unique : DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED,
                t->gate_offset + e * t->gate_expert_bytes, t->up_offset + e * t->gate_expert_bytes,
                t->down_offset + e * t->down_expert_bytes, t->gate_expert_bytes, t->down_expert_bytes);
            if (!entry[i]) missing[n_missing++] = unique[i];
        }
        if (n_missing == 0) break;
        if (attempt == 4) {
            ok = 0;
            break;
        }
        ds4_gpu_stream_expert_pending_load_clear();
        for (uint32_t at = 0; ok && at < n_missing; at += DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED) {
            const uint32_t n = n_missing - at < DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED ?
                n_missing - at : DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED;
            ok = ds4_gpu_stream_expert_cache_begin_selected_load(t, missing + at, n);
            ds4_gpu_stream_expert_pending_load_clear();
        }
    }
    for (uint32_t i = 0; ok && i < n_unique; i++) {
        ok = entry[i]->slab_backed &&
             ds4_gpu_stream_expert_cache_set_addr_slot_raw(t->layer, (uint32_t)unique[i],
                 entry[i]->gate_buffer, entry[i]->gate_inner, entry[i]->up_buffer, entry[i]->up_inner,
                 entry[i]->down_buffer, entry[i]->down_inner);
    }
    if (!ok) {
        qwen4_handshake_fail("an expert could not be loaded into the cache");
        id<MTLBuffer> any = g_stream_expert_cache_slabs[0];
        for (uint32_t i = 0; i < h->n_ids; i++) {
            ids[i] = h->ids[i];
            if (ids[i] >= 0 && (uint32_t)ids[i] < t->n_total_expert)
                (void)ds4_gpu_stream_expert_cache_set_addr_slot_raw(t->layer, (uint32_t)ids[i],
                                                                    any, 0, any, 0, any, 0);
        }
    }
    g_qwen4_hs_event.signaledValue = h->value + 1u;
    g_qwen4_hs_stats.service_ms += ds4_gpu_now_ms() - t1;
}

/* Services the pending layers except the newest `keep`. */
static void qwen4_handshake_service_pending(uint32_t keep) {
    if (g_qwen4_hs_servicing || g_qwen4_hs_n_pending <= keep) return;
    g_qwen4_hs_servicing = 1;
    const uint32_t n = g_qwen4_hs_n_pending - keep;
    for (uint32_t i = 0; i < n; i++) qwen4_handshake_service(&g_qwen4_hs_pending[i]);
    for (uint32_t i = 0; i < keep; i++) g_qwen4_hs_pending[i] = g_qwen4_hs_pending[n + i];
    g_qwen4_hs_n_pending = keep;
    g_qwen4_hs_servicing = 0;
}

/* Called before every command-buffer wait. Returns 0 once after a failure. */
static int qwen4_stream_handshake_service_all(void) {
    if (g_qwen4_hs_n_pending) qwen4_handshake_service_pending(0);
    if (!g_qwen4_hs_failed || g_qwen4_hs_servicing) return 1;
    g_qwen4_hs_failed = 0;
    return 0;
}

/* Allocates the rest of the expert cache's slabs. Kernels encoded before a
 * load reach every slab, so none may appear after encoding; the new slots go
 * to the free list unlocked and are locked when first used, as before. */
static int qwen4_handshake_reserve_slabs(uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    if (budget == 0 || budget > DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES) return 0;
    uint64_t slot_bytes = gate_expert_bytes * 2ull + down_expert_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    if (page) slot_bytes = round_up_u64(slot_bytes, page);
    if (slot_bytes == 0 || (g_stream_expert_cache_slab_slot_bytes &&
                            g_stream_expert_cache_slab_slot_bytes != slot_bytes)) return 0;
    if (g_stream_expert_cache_slab_total_slots >= budget) return 1;
    /* Slots are taken from the free list or the newest slab only, so hand
     * the newest slab's unused slots to the free list first. */
    if (g_stream_expert_cache_slab_count) {
        const uint32_t last = g_stream_expert_cache_slab_count - 1u;
        while (g_stream_expert_cache_slab_slots_used[last] < g_stream_expert_cache_slab_slot_count[last])
            ds4_gpu_stream_expert_slab_push_free_slot(g_stream_expert_cache_slab_start_slot[last] +
                                                      g_stream_expert_cache_slab_slots_used[last]++);
    }
    while (g_stream_expert_cache_slab_total_slots < budget) {
        if (g_stream_expert_cache_slab_count >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS) return 0;
        uint64_t slots = ds4_gpu_stream_expert_slab_target_bytes() / slot_bytes;
        if (slots == 0) slots = 1;
        if (slots > budget - g_stream_expert_cache_slab_total_slots)
            slots = budget - g_stream_expert_cache_slab_total_slots;
        id<MTLBuffer> slab = nil;
        for (; slots && !slab; slots /= 2u) {
            if (slots * slot_bytes <= (uint64_t)NSUIntegerMax)
                slab = ds4_gpu_stream_expert_alloc_slab_buffer(slots * slot_bytes, @"ds4_stream_expert_slab");
            if (slab) break;
        }
        if (!slab) return 0;
        const uint32_t i = g_stream_expert_cache_slab_count++;
        g_stream_expert_cache_slabs[i] = slab;
        g_stream_expert_cache_slab_start_slot[i] = g_stream_expert_cache_slab_total_slots;
        g_stream_expert_cache_slab_slot_count[i] = (uint32_t)slots;
        g_stream_expert_cache_slab_slots_used[i] = (uint32_t)slots;
        g_stream_expert_cache_slab_slot_bytes = slot_bytes;
        for (uint32_t k = 0; k < (uint32_t)slots; k++)
            ds4_gpu_stream_expert_slab_push_free_slot(g_stream_expert_cache_slab_total_slots + k);
        g_stream_expert_cache_slab_total_slots += (uint32_t)slots;
    }
    return 1;
}

/* Encodes one streamed decode or verify layer (up to three tokens) with the
 * handshake. Returns 0 when
 * the layer does not qualify (the ordinary path runs), else 1 with *ok set. */
static int qwen4_stream_handshake_layer(
        int *ok, int had_batch, const ds4_gpu_stream_expert_table *table,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *part, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *selected, uint32_t gate_type, uint32_t down_type,
        uint32_t n_tokens, uint32_t n_slots, uint32_t in_dim, uint32_t ff_dim,
        uint64_t shared_gate_offset, uint64_t shared_up_offset, uint64_t shared_down_offset,
        uint32_t shared_type, uint32_t shared_down_type) {
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    if (!qwen4_stream_handshake_enabled() || !had_batch || !g_batch_cb ||
        n_tokens == 0 || n_tokens > QWEN4_HS_MAX_TOKENS ||
        n_slots > DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED || gate_type == 12u ||
        !ds4_gpu_stream_expert_slab_enabled() ||
        !ds4_gpu_stream_expert_cache_note_expert_size(table->gate_expert_bytes, table->down_expert_bytes) ||
        (g_stream_expert_cache_slab_total_slots < budget &&
         !qwen4_handshake_reserve_slabs(table->gate_expert_bytes, table->down_expert_bytes)) ||
        budget == 0 || g_stream_expert_cache_slab_count == 0 ||
        ds4_gpu_stream_expert_cache_effective_cap(table->layer, table->n_total_expert, n_slots) == 0 ||
        !ds4_gpu_tensor_contents((ds4_gpu_tensor *)selected)) {
        if (qwen4_stream_handshake_enabled()) {
            g_qwen4_hs_stats.ordinary++;
            g_qwen4_hs_stats.not_full += budget == 0 || g_stream_expert_cache_slab_total_slots < budget;
        }
        g_qwen4_hs_dirty = 1;
        return 0;
    }
    if (!g_qwen4_hs_event) {
        g_qwen4_hs_event = [g_device newSharedEvent];
        if (!g_qwen4_hs_event) {
            g_qwen4_hs_dirty = 1;
            return 0;
        }
        g_qwen4_hs_value = g_qwen4_hs_event.signaledValue;
    }
    if (g_qwen4_hs_dirty) {
        /* Join ordinary layers' in-flight experts before any handshake load. */
        if (!ds4_gpu_synchronize() || (!g_batch_cb && !ds4_gpu_begin_commands())) {
            *ok = 0;
            return 1;
        }
        g_qwen4_hs_dirty = 0;
        g_qwen4_hs_stats.drains++;
    }
    if (!ds4_gpu_stream_expert_cache_ensure_addr_buffers(table->layer)) {
        *ok = 0;
        return 1;
    }

    if (g_qwen4_hs_n_pending >= 2u) qwen4_handshake_service_pending(0);
    ds4_gpu_close_batch_encoder();
    const uint64_t value = ++g_qwen4_hs_value;
    ++g_qwen4_hs_value;
    [g_batch_cb encodeSignalEvent:g_qwen4_hs_event value:value];
    [g_batch_cb encodeWaitForEvent:g_qwen4_hs_event value:value + 1u];
    g_batch_has_work = YES;
    g_qwen4_hs_stats.layers++;
    /* Recorded before anything else can fail: the GPU now waits for it. */
    g_qwen4_hs_pending[g_qwen4_hs_n_pending++] = (qwen4_handshake_layer) {
        .table = *table, .value = value,
        .ids = (const volatile int32_t *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)selected),
        .n_ids = n_tokens * n_slots,
    };

    qwen4_stream_weights stream = {
        .table = table,
        .gate = g_stream_expert_cache_gate_addr_buffers[table->layer],
        .up = g_stream_expert_cache_up_addr_buffers[table->layer],
        .down = g_stream_expert_cache_down_addr_buffers[table->layer],
        .pending = [NSSet setWithObjects:g_stream_expert_cache_slabs count:g_stream_expert_cache_slab_count],
        .addresses = true,
    };
    g_qwen4_stream_weights = &stream;
    *ok = ds4_gpu_qwen4_moe_mid_tensor(mid, x, selected, table->model_map, table->model_size,
            table->gate_offset, table->up_offset, gate_type, table->n_total_expert,
            n_tokens, n_slots, in_dim, ff_dim, shared_gate_offset, shared_up_offset, shared_type) &&
          ds4_gpu_qwen4_moe_down_tensor(part, mid, selected, table->model_map, table->model_size,
            table->down_offset, down_type, table->n_total_expert,
            n_tokens, n_slots, ff_dim, in_dim, shared_down_offset, shared_down_type) &&
          ds4_gpu_flush_commands();
    g_qwen4_stream_weights = NULL;
    /* Layer l is queued; service layer l - 1 while the GPU runs up to l. */
    qwen4_handshake_service_pending(1);
    if (g_qwen4_hs_failed) *ok = 0;
    return 1;
}
