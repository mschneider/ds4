/* Ternary Bonsai 2: model contract and weight binding (fork-owned).
 * Textually included into ds4.c just before config_validate_model, after the
 * GGUF metadata helpers and the Qwen3.8 validator it borrows from.
 *
 * The GGUF is PrismML's PQ2_0 release used as is (arch "qwen35").  Every
 * metadata key the graph depends on is checked against DS4_SHAPE_BONSAI2,
 * including the Hadamard folding contract: running a folded weight without
 * its activation transform would produce fluent-looking garbage, so a file
 * whose contract differs in any way is refused at load. */

/* prism.hadamard.sign_values split by input width, in sign_widths order. */
enum {
    BONSAI2_SIGN_EMBD = 0,  /* 5120: everything fed by a normed hidden state */
    BONSAI2_SIGN_INNER,     /* 6144: ssm_out and attn_output */
    BONSAI2_SIGN_FF,        /* 17408: ffn_down */
    BONSAI2_SIGN_COUNT
};

static uint32_t g_bonsai2_sign_width[BONSAI2_SIGN_COUNT];
static float *g_bonsai2_signs[BONSAI2_SIGN_COUNT];

static uint32_t bonsai2_sign_index(uint64_t width) {
    for (uint32_t i = 0; i < BONSAI2_SIGN_COUNT; i++) {
        if (g_bonsai2_sign_width[i] == width) return i;
    }
    return BONSAI2_SIGN_COUNT;
}

static void bonsai2_expect_string(const ds4_model *m, const char *key, const char *expected) {
    ds4_str v = {0};
    if (!model_get_string(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    if (!ds4_streq(v, expected)) {
        fprintf(stderr, "ds4: %s is '%.*s', expected '%s'\n", key, (int)v.len, v.ptr, expected);
        exit(1);
    }
}

/* Reads a string array and marks every named tensor in `seen` (indexed like
 * m->tensors).  Unknown or repeated names are fatal. */
static uint64_t bonsai2_mark_names(const ds4_model *m, const char *key, uint8_t *seen) {
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) || arr.type != GGUF_VALUE_STRING) {
        fprintf(stderr, "ds4: required string array metadata key is missing: %s\n", key);
        exit(1);
    }
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < arr.len; i++) {
        ds4_str name;
        if (!cursor_string(&c, &name)) ds4_die(c.error);
        uint64_t ti = 0;
        while (ti < m->n_tensors && !ds4_str_eq(m->tensors[ti].name, name)) ti++;
        if (ti == m->n_tensors || seen[ti]) {
            fprintf(stderr, "ds4: %s lists %s tensor %.*s\n", key,
                    ti == m->n_tensors ? "unknown" : "repeated", (int)name.len, name.ptr);
            exit(1);
        }
        seen[ti] = 1;
    }
    return arr.len;
}

/* Folding contract version 1: explicit +-1 signs per input width, a 1024
 * block normalized Sylvester-Walsh-Hadamard on the input axis, every PQ2_0
 * weight folded except the latent embedding, which is the only inverse, and
 * GDN ssm_out columns left in grouped head order (the graph permutes its
 * input).  Signs stay on the host until a graph uploads them. */
static void bonsai2_validate_hadamard(const ds4_model *m) {
    config_expect_u32("prism.hadamard.version", required_u32(m, "prism.hadamard.version"), 1);
    config_expect_u32("prism.hadamard.block_size", required_u32(m, "prism.hadamard.block_size"), 1024);
    bonsai2_expect_string(m, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
    bonsai2_expect_string(m, "prism.hadamard.axis", "input-last-dimension");
    bonsai2_expect_string(m, "prism.hadamard.sign_mode", "explicit");
    config_expect_bool("prism.hadamard.gdn_v_grouped", required_bool(m, "prism.hadamard.gdn_v_grouped"), true);
    bool tied = false;
    if (model_get_bool(m, "prism.hadamard.tied_output", &tied) && tied) {
        ds4_die("tied Bonsai output heads are not supported");
    }

    static const uint32_t widths[BONSAI2_SIGN_COUNT] = { 5120, 6144, 17408 };
    uint64_t got[BONSAI2_SIGN_COUNT];
    uint32_t n = 0;
    config_read_qwen4_u64_array(m, "prism.hadamard.sign_widths", got, BONSAI2_SIGN_COUNT, &n);
    if (n != BONSAI2_SIGN_COUNT) ds4_die("prism.hadamard.sign_widths must list three widths");
    uint64_t total = 0;
    for (uint32_t i = 0; i < BONSAI2_SIGN_COUNT; i++) {
        config_expect_u64("prism.hadamard.sign_widths", got[i], widths[i]);
        g_bonsai2_sign_width[i] = widths[i];
        total += widths[i];
    }
    ds4_array_ref arr;
    if (!model_get_array(m, "prism.hadamard.sign_values", &arr) || arr.type != GGUF_VALUE_INT32 ||
        arr.len != total) {
        ds4_die("prism.hadamard.sign_values must hold one int32 per sign width entry");
    }
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < BONSAI2_SIGN_COUNT; i++) {
        free(g_bonsai2_signs[i]);
        g_bonsai2_signs[i] = xmalloc(widths[i] * sizeof(float));
        for (uint32_t j = 0; j < widths[i]; j++) {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if (v != 1 && v != -1) ds4_die("prism.hadamard.sign_values must be +1 or -1");
            g_bonsai2_signs[i][j] = (float)v;
        }
    }

    uint8_t *folded = xcalloc((size_t)m->n_tensors, 1);
    uint8_t *inverse = xcalloc((size_t)m->n_tensors, 1);
    bonsai2_mark_names(m, "prism.hadamard.weight_names", folded);
    if (bonsai2_mark_names(m, "prism.hadamard.inverse_weight_names", inverse) != 1) {
        ds4_die("prism.hadamard.inverse_weight_names must name only token_embd.weight");
    }
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        const bool embd = ds4_streq(t->name, "token_embd.weight");
        const bool pq2 = t->type == DS4_TENSOR_PQ2_0;
        if (inverse[i] != embd || folded[i] != (pq2 && !embd) ||
            (pq2 && bonsai2_sign_index(t->dim[0]) == BONSAI2_SIGN_COUNT)) {
            fprintf(stderr, "ds4: tensor %.*s does not match the Hadamard folding contract\n",
                    (int)t->name.len, t->name.ptr);
            exit(1);
        }
    }
    free(inverse);
    free(folded);
}

static void config_validate_bonsai2_model(const ds4_model *m) {
    config_expect_u32("embedding_length", required_u32(m, "qwen35.embedding_length"),
                      DS4_SHAPE_BONSAI2.n_embd);
    g_ds4_shape = DS4_SHAPE_BONSAI2;
    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));
    memset(&g_ds4_qwen4_ple, 0, sizeof(g_ds4_qwen4_ple));

    uint32_t nextn = 0;
    if (model_get_u32(m, "qwen35.nextn_predict_layers", &nextn) && nextn != 0) {
        ds4_die("Bonsai MTP layers are not supported");
    }
    config_expect_u32("block_count", required_u32(m, "qwen35.block_count"), DS4_N_LAYER);
    config_expect_u32("feed_forward_length", required_u32(m, "qwen35.feed_forward_length"), DS4_N_FF_DENSE);
    config_expect_u32("attention.head_count", required_u32(m, "qwen35.attention.head_count"), DS4_N_HEAD);
    config_expect_u32("attention.head_count_kv",
                      required_u32(m, "qwen35.attention.head_count_kv"), DS4_N_HEAD_KV);
    config_expect_u32("attention.key_length", required_u32(m, "qwen35.attention.key_length"), DS4_N_HEAD_DIM);
    config_expect_u32("attention.value_length",
                      required_u32(m, "qwen35.attention.value_length"), DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count", required_u32(m, "qwen35.rope.dimension_count"), DS4_N_ROT);
    config_expect_f32("rope.freq_base", required_f32(m, "qwen35.rope.freq_base"), DS4_ROPE_FREQ_BASE);
    /* Interleaved MRoPE over the 32 rotated pairs, the layout the Qwen3.8
     * attention kernels implement; text positions repeat in every section. */
    {
        uint64_t sections[4];
        uint32_t n = 0;
        static const uint64_t expected[4] = { 11, 11, 10, 0 };
        config_read_qwen4_u64_array(m, "qwen35.rope.dimension_sections", sections, 4, &n);
        if (n != 4) ds4_die("qwen35.rope.dimension_sections must have four entries");
        for (uint32_t i = 0; i < 4; i++) config_expect_u64("rope.dimension_sections", sections[i], expected[i]);
    }
    qwen4_rope_configure(required_u32(m, "qwen35.context_length"));
    config_expect_epsilon("attention.layer_norm_rms_epsilon",
                          required_f32(m, "qwen35.attention.layer_norm_rms_epsilon"), DS4_RMS_EPS);
    config_expect_u32("ssm.conv_kernel", required_u32(m, "qwen35.ssm.conv_kernel"), DS4_N_LIN_CONV);
    config_expect_u32("ssm.state_size", required_u32(m, "qwen35.ssm.state_size"), DS4_N_LIN_HEAD_DIM);
    config_expect_u32("ssm.group_count", required_u32(m, "qwen35.ssm.group_count"), DS4_N_LIN_K_HEAD);
    config_expect_u32("ssm.time_step_rank", required_u32(m, "qwen35.ssm.time_step_rank"), DS4_N_LIN_V_HEAD);
    config_expect_u32("ssm.inner_size", required_u32(m, "qwen35.ssm.inner_size"),
                      DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM);
    config_expect_u32("full_attention_interval",
                      required_u32(m, "qwen35.full_attention_interval"), DS4_N_FULL_ATTN_INTERVAL);
    bonsai2_validate_hadamard(m);
}

static void weights_bind_bonsai2_output(ds4_weights *w, const ds4_model *m, bool required, bool optional) {
    if (required) {
        w->output_norm = required_tensor(m, "output_norm.weight");
        w->output      = required_tensor(m, "output.weight");
    } else if (optional) {
        w->output_norm = model_find_tensor(m, "output_norm.weight");
        w->output      = model_find_tensor(m, "output.weight");
    }
}

static void weights_bind_bonsai2_layer(ds4_layer_weights *l, const ds4_model *m, uint32_t il) {
    l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    if (ds4_qwen4_layer_is_linear(il)) {
        l->lin_qkv     = required_tensorf(m, "blk.%u.attn_qkv.weight", il);
        l->lin_gate    = required_tensorf(m, "blk.%u.attn_gate.weight", il);
        l->lin_conv    = required_tensorf(m, "blk.%u.ssm_conv1d.weight", il);
        l->lin_dt_bias = required_tensorf(m, "blk.%u.ssm_dt.bias", il);
        l->lin_a       = required_tensorf(m, "blk.%u.ssm_a", il);
        l->lin_beta    = required_tensorf(m, "blk.%u.ssm_beta.weight", il);
        l->lin_alpha   = required_tensorf(m, "blk.%u.ssm_alpha.weight", il);
        l->lin_norm    = required_tensorf(m, "blk.%u.ssm_norm.weight", il);
        l->lin_out     = required_tensorf(m, "blk.%u.ssm_out.weight", il);
    } else {
        l->attn_q      = required_tensorf(m, "blk.%u.attn_q.weight", il);
        l->attn_k      = required_tensorf(m, "blk.%u.attn_k.weight", il);
        l->attn_v      = required_tensorf(m, "blk.%u.attn_v.weight", il);
        l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
        l->attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
        l->attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
    }
    l->ffn_norm = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
    l->ffn_gate = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
    l->ffn_up   = required_tensorf(m, "blk.%u.ffn_up.weight", il);
    l->ffn_down = required_tensorf(m, "blk.%u.ffn_down.weight", il);
}

static void bonsai2_expect_pq2(const ds4_tensor *t, uint64_t in_dim, uint64_t out_dim) {
    tensor_expect_layout(t, DS4_TENSOR_PQ2_0, 2, in_dim, out_dim, 0);
}

static void bonsai2_expect_f32(const ds4_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1) {
    tensor_expect_layout(t, DS4_TENSOR_F32, ndim, d0, d1, 0);
}

static void weights_validate_bonsai2_layout(
        const ds4_weights *w, uint32_t layer_start, uint32_t layer_end,
        bool require_token_embd, bool require_output) {
    const uint64_t E = DS4_N_EMBD, F = DS4_N_FF_DENSE, V = DS4_N_VOCAB;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    if (require_token_embd) bonsai2_expect_pq2(w->token_embd, E, V);
    if (require_output) {
        bonsai2_expect_f32(w->output_norm, 1, E, 0);
        bonsai2_expect_pq2(w->output, E, V);
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        bonsai2_expect_f32(l->attn_norm, 1, E, 0);
        bonsai2_expect_f32(l->ffn_norm, 1, E, 0);
        bonsai2_expect_pq2(l->ffn_gate, E, F);
        bonsai2_expect_pq2(l->ffn_up, E, F);
        bonsai2_expect_pq2(l->ffn_down, F, E);
        if (ds4_qwen4_layer_is_linear(il)) {
            bonsai2_expect_pq2(l->lin_qkv, E, DS4_N_LIN_CONV_DIM);
            bonsai2_expect_pq2(l->lin_gate, E, v_dim);
            bonsai2_expect_pq2(l->lin_out, v_dim, E);
            tensor_expect_layout(l->lin_alpha, DS4_TENSOR_BF16, 2, E, DS4_N_LIN_V_HEAD, 0);
            tensor_expect_layout(l->lin_beta, DS4_TENSOR_BF16, 2, E, DS4_N_LIN_V_HEAD, 0);
            bonsai2_expect_f32(l->lin_conv, 2, DS4_N_LIN_CONV, DS4_N_LIN_CONV_DIM);
            bonsai2_expect_f32(l->lin_a, 1, DS4_N_LIN_V_HEAD, 0);
            bonsai2_expect_f32(l->lin_dt_bias, 1, DS4_N_LIN_V_HEAD, 0);
            bonsai2_expect_f32(l->lin_norm, 1, DS4_N_LIN_HEAD_DIM, 0);
        } else {
            /* attn_q rows hold every head's query followed by its output gate */
            bonsai2_expect_pq2(l->attn_q, E, 2u * q_dim);
            bonsai2_expect_pq2(l->attn_k, E, kv_dim);
            bonsai2_expect_pq2(l->attn_v, E, kv_dim);
            bonsai2_expect_pq2(l->attn_output, q_dim, E);
            bonsai2_expect_f32(l->attn_q_norm, 1, DS4_N_HEAD_DIM, 0);
            bonsai2_expect_f32(l->attn_k_norm, 1, DS4_N_HEAD_DIM, 0);
        }
    }
}

/* Host dequantization of one PQ2_0 row: 34-byte blocks of an fp16 scale d and
 * 128 two-bit codes, least significant pair first, value (code - 1) * d. */
static void bonsai2_pq2_row(const ds4_model *m, const ds4_tensor *t, uint64_t row, float *out) {
    const uint64_t blocks = t->dim[0] / 128u;
    const uint8_t *p = (const uint8_t *)tensor_data(m, t) + row * blocks * 34u;
    for (uint64_t b = 0; b < blocks; b++, p += 34u) {
        uint16_t dh;
        memcpy(&dh, p, sizeof(dh));
        const float d = f16_to_f32(dh);
        for (uint32_t j = 0; j < 128u; j++) {
            out[b * 128u + j] = (float)((int)((p[2 + j / 4u] >> (2u * (j % 4u))) & 3u) - 1) * d;
        }
    }
}

/* Bonsai runs only on the resident single-device Metal graph for now; the
 * remaining Qwen3.8 restrictions (TP, SSD streaming, ...) are checked after
 * this by the shared Qwen3.8 gate. */
static bool bonsai2_engine_supported(const ds4_engine_options *opt, ds4_backend backend) {
    const char *why = NULL;
    if (backend != DS4_BACKEND_METAL) why = "needs the Metal backend";
    else if (opt->first_token_test || opt->metal_graph_test) why = "has no CPU reference diagnostics";
    else if (opt->glm_mtp || (opt->mtp_path && opt->mtp_path[0])) why = "has no MTP weights";
    if (why) fprintf(stderr, "ds4: Ternary Bonsai 2 %s\n", why);
    return why == NULL;
}

bool ds4_engine_is_bonsai2(ds4_engine *e) {
    (void)e;
    return ds4_model_is_bonsai2();
}

#ifdef DS4_HAS_QWEN4_GPU
/* Defined with the graph in ds4_bonsai2.inc; the Qwen3.8 graph entry points
 * hand Bonsai sessions over to them. */
struct ds4_qwen4_gpu_graph;
static bool bonsai2_graph_alloc(struct ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens,
                                const struct ds4_qwen4_gpu_graph *shared,
                                ds4_gpu_tensor *const *state_pool, ds4_gpu_tensor *const *hist_pool,
                                uint32_t slot);
static bool bonsai2_graph_forward_tokens(struct ds4_qwen4_gpu_graph *g, const ds4_model *m,
                                         const ds4_weights *w, const int *tokens, uint32_t T,
                                         float *logits_out, bool all_rows);
#endif
