/* Ternary Bonsai 2 27B (fork-owned), textually included into ds4.c right
 * after the Qwen3.8 layer predicates.
 *
 * PrismML's ternary Qwen3.8-27B: 64 dense layers repeating 3 Gated DeltaNet
 * + 1 gated GQA attention, SwiGLU FFNs, untied embedding and head.  The GDN
 * and attention geometry matches Qwen3.8 Flash Next, so Bonsai runs as a
 * variant of that family: sessions, payloads, tokenizer and chat template are
 * shared, and only the graph differs (no hyper-connections, n-grams, MoE,
 * indexer or MTP; every projection is PQ2_0 behind a Hadamard rotation).
 *
 * The Qwen3.8 bookkeeping that does not apply is set to inert values rather
 * than branched around: one residual stream (n_hc = 1), no PLE layer (index
 * past the last layer, conv width 1 so its history is empty) and no indexer
 * (zero-width keys, so the payload carries empty spans for them). */
static const ds4_shape DS4_SHAPE_BONSAI2 = {
    .name = "Ternary Bonsai 2 27B",
    .family = DS4_MODEL_FAMILY_QWEN4_EXP,
    .variant = DS4_VARIANT_BONSAI2,
    .n_layer = 64,
    .n_embd = 5120,
    .n_vocab = 248320,
    .n_head = 24,
    .n_head_kv = 4,
    .n_head_dim = 256,
    .n_value_dim = 256,
    .n_rot = 64,
    .n_ff_dense = 17408,
    .n_hc = 1,
    .n_lin_k_head = 16,
    .n_lin_v_head = 48,
    .n_lin_head_dim = 128,
    .n_lin_conv = 4,
    .n_full_attn_interval = 4,
    .n_ple_layer = DS4_MAX_LAYER,
    .n_ple_conv = 1,
    .ple_eos_id = 248044,
    .rms_eps = 1.0e-6f,
    .rope_freq_base = 10000000.0f,
    .rope_orig_ctx = 262144,
};

static bool ds4_model_is_bonsai2(void) {
    return DS4_MODEL_VARIANT == DS4_VARIANT_BONSAI2;
}
