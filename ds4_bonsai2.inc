/* Ternary Bonsai 2 GPU graph (fork-owned), textually included into ds4.c
 * right after the Qwen3.8 GPU graph section, so the Qwen3.8 graph type, its
 * GDN/attention kernel wrappers and helpers are visible here.
 *
 * A Bonsai session is an ordinary Qwen3.8 session whose graph allocation and
 * forward pass dispatch here (qwen4_graph_alloc / qwen4_graph_forward_tokens),
 * so recurrent state, KV caches, positions, snapshots and disk payloads use
 * the Qwen3.8 fields and code unchanged.  The graph itself is the plain
 * Qwen3.5-style decoder:
 *
 *   h = s * H(embd_latent[token])                      latent row, inverse rotation
 *   per layer:  x = rms(h) * attn_norm
 *               h += mixer(x)                          GDN or gated attention
 *               x = rms(h) * post_attention_norm
 *               h += down(silu(gate(x)) * up(x))
 *   logits = output(rms(h) * output_norm)
 *
 * Every projection is PQ2_0 with a folded Hadamard rotation, so its input is
 * rotated once (sign, then blockwise FWHT) and shared by all projections that
 * read it: qkv/z or q/k/v off the normed hidden state, gate/up off the FFN
 * input.  Only the BF16 GDN alpha/beta projections read the unrotated normed
 * state (g->xn).  Buffer roles: R residual, xn normed, mixed rotated normed,
 * blk mixer/FFN output, q doubles as the rotated 6144-wide input of ssm_out
 * and attn_output, sh_gate/sh_up/sh_mid the FFN. */

#ifdef DS4_HAS_QWEN4_GPU
#ifdef DS4_HAS_QWEN4_METAL
#include "ds4_bonsai2_gpu.h"

/* Uploaded once from the validated contract and shared by every session of
 * the process; the model (and so the signs) cannot change underneath them. */
static ds4_gpu_tensor *g_bonsai2_sign_gpu[BONSAI2_SIGN_COUNT];

static bool bonsai2_signs_ready(void) {
    for (uint32_t i = 0; i < BONSAI2_SIGN_COUNT; i++) {
        if (g_bonsai2_sign_gpu[i]) continue;
        const uint64_t bytes = (uint64_t)g_bonsai2_sign_width[i] * sizeof(float);
        if (!g_bonsai2_signs[i]) return false;
        g_bonsai2_sign_gpu[i] = ds4_gpu_tensor_alloc(bytes);
        if (!g_bonsai2_sign_gpu[i] ||
            !ds4_gpu_tensor_write(g_bonsai2_sign_gpu[i], 0, g_bonsai2_signs[i], bytes)) {
            ds4_gpu_tensor_free(g_bonsai2_sign_gpu[i]);
            g_bonsai2_sign_gpu[i] = NULL;
            return false;
        }
    }
    return true;
}


static bool bonsai2_mm(ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w,
                       const ds4_gpu_tensor *x, uint32_t T) {
    if (ds4_gpu_bonsai2_matmul_pq2_tensor(out, m->map, m->size, w->abs_offset,
                                          (uint32_t)w->dim[0], (uint32_t)w->dim[1], x, T)) {
        return true;
    }
    fprintf(stderr, "ds4: Bonsai PQ2_0 matmul failed for %.*s (%u tokens)\n",
            (int)w->name.len, w->name.ptr, T);
    return false;
}

/* Normed hidden state, rotated for the PQ2_0 projections; xn (optional)
 * keeps the unrotated copy for the BF16 GDN gates. */
static bool bonsai2_norm(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_tensor *norm,
                         const ds4_gpu_tensor *x, ds4_gpu_tensor *xn, uint32_t T) {
    return ds4_gpu_bonsai2_rms_rotate_tensor(g->mixed, xn, x, m->map, m->size, norm->abs_offset,
                                             g_bonsai2_sign_gpu[BONSAI2_SIGN_EMBD], T, DS4_N_EMBD,
                                             DS4_RMS_EPS) != 0;
}

static ds4_gpu_tensor *bonsai2_alloc_f32(uint64_t n) {
    return qwen4_graph_alloc_f32(n ? n : 1u);
}

static bool bonsai2_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens,
                                const ds4_qwen4_gpu_graph *shared,
                                ds4_gpu_tensor *const *state_pool, ds4_gpu_tensor *const *hist_pool,
                                uint32_t slot) {
    memset(g, 0, sizeof(*g));
    if (!bonsai2_signs_ready()) {
        fprintf(stderr, "ds4: Bonsai Hadamard signs are not available\n");
        return false;
    }
    const uint64_t E = DS4_N_EMBD, T = cap_tokens, F = DS4_N_FF_DENSE;
    const uint64_t conv_dim = DS4_N_LIN_CONV_DIM;
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    g->ctx_cap = ctx_cap;
    g->cap_tokens = cap_tokens;
    g->n_logit_rows = 1;
    /* the session arena compares block capacities; Bonsai has no blocks but
     * keeps the Qwen3.8 value so arena reuse follows the same rule */
    g->n_block_cap = ctx_cap / 4u + 1u;

    bool ok = true;
    g->owns_scratch = shared == NULL;
#define BONSAI2_ALLOC(field_, n_) do { g->field_ = bonsai2_alloc_f32(n_); ok = ok && g->field_; } while (0)
    if (shared) {
#define BONSAI2_BORROW(field_) g->field_ = shared->field_;
        DS4_QWEN4_SCRATCH_FIELDS(BONSAI2_BORROW)
#undef BONSAI2_BORROW
        g->host_row = shared->host_row;
    } else {
        BONSAI2_ALLOC(R, T * E);
        BONSAI2_ALLOC(xn, T * E);
        BONSAI2_ALLOC(mixed, T * E);
        BONSAI2_ALLOC(blk, T * E);
        BONSAI2_ALLOC(qkv, T * conv_dim);
        BONSAI2_ALLOC(z, T * v_dim);
        BONSAI2_ALLOC(ga, T * DS4_N_LIN_V_HEAD);
        BONSAI2_ALLOC(gb, T * DS4_N_LIN_V_HEAD);
        BONSAI2_ALLOC(lin_o, T * v_dim);
        BONSAI2_ALLOC(qg, T * 2u * q_dim);
        BONSAI2_ALLOC(kp, T * kv_dim);
        BONSAI2_ALLOC(vp, T * kv_dim);
        BONSAI2_ALLOC(q, T * (q_dim > v_dim ? q_dim : v_dim));
        BONSAI2_ALLOC(gate, T * q_dim);
        BONSAI2_ALLOC(attn_o, T * q_dim);
        BONSAI2_ALLOC(attn_part, ds4_gpu_qwen4_attn_part_floats(3u, DS4_N_HEAD, DS4_N_HEAD_DIM));
        BONSAI2_ALLOC(sh_gate, T * F);
        BONSAI2_ALLOC(sh_up, T * F);
        BONSAI2_ALLOC(sh_mid, T * F);
        g->host_row = xmalloc(T * E * sizeof(float));
    }
    /* Inert Qwen3.8 state: the payload and reset code walk these with zero
     * sizes (no PLE history, zero-width indexer keys) but expect tensors. */
    BONSAI2_ALLOC(ple_hist, 1);
    BONSAI2_ALLOC(logits, DS4_N_VOCAB);
    const uint64_t state_bytes = v_dim * DS4_N_LIN_HEAD_DIM * sizeof(float);
    const uint64_t hist_bytes = (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim * sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        if (ds4_qwen4_layer_is_linear(il)) {
            if (state_pool && hist_pool) {
                g->layer_lin_state[il] = ds4_gpu_tensor_view(state_pool[il], (uint64_t)slot * state_bytes,
                                                             state_bytes);
                g->layer_lin_hist[il] = ds4_gpu_tensor_view(hist_pool[il], (uint64_t)slot * hist_bytes,
                                                            hist_bytes);
            } else {
                g->layer_lin_state[il] = bonsai2_alloc_f32(v_dim * DS4_N_LIN_HEAD_DIM);
                g->layer_lin_hist[il] = bonsai2_alloc_f32((uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim);
            }
            ok = g->layer_lin_state[il] && g->layer_lin_hist[il];
        } else {
            g->layer_k_cache[il] = ds4_gpu_tensor_alloc((uint64_t)ctx_cap * kv_dim * 2u);
            g->layer_v_cache[il] = ds4_gpu_tensor_alloc((uint64_t)ctx_cap * kv_dim * 2u);
            BONSAI2_ALLOC(layer_ik_cache[il], 1);
            BONSAI2_ALLOC(layer_block_key[il], 1);
            ok = ok && g->layer_k_cache[il] && g->layer_v_cache[il];
        }
    }
#undef BONSAI2_ALLOC
    g->pos3 = bonsai2_alloc_f32((uint64_t)ctx_cap * 4u);
    ok = ok && g->pos3;
    g->host_pos3 = xmalloc(T * 4u * sizeof(uint32_t));
    g->host_logits = xmalloc((uint64_t)DS4_N_VOCAB * sizeof(float));
    if (!ok) {
        fprintf(stderr, "ds4: Bonsai graph allocation failed (ctx %u)\n", ctx_cap);
        qwen4_graph_free(g);
        return false;
    }
    return true;
}

/* z = H(s * h) on the host for one hidden row: the forward rotation, so
 * that the graph's embedding unrotate (h = s * H z) maps it back. */
static void bonsai2_rotate_row_host(float *z, const float *h) {
    const float *s = g_bonsai2_signs[BONSAI2_SIGN_EMBD];
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) z[i] = h[i] * s[i];
    for (uint32_t b = 0; b < DS4_N_EMBD; b += DS4_BONSAI2_HADAMARD_BLOCK) {
        float *v = z + b;
        for (uint32_t len = 1; len < DS4_BONSAI2_HADAMARD_BLOCK; len <<= 1) {
            for (uint32_t i = 0; i < DS4_BONSAI2_HADAMARD_BLOCK; i += len << 1) {
                for (uint32_t j = i; j < i + len; j++) {
                    const float a = v[j], c = v[j + len];
                    v[j] = a + c;
                    v[j + len] = a - c;
                }
            }
        }
        for (uint32_t i = 0; i < DS4_BONSAI2_HADAMARD_BLOCK; i++) v[i] *= 1.0f / 32.0f;
    }
}

/* Token rows and rope positions for T tokens at g->pos.  Text rows are the
 * latent (rotated) PQ2_0 embedding rows, un-rotated on the GPU for the whole
 * batch.  Image rows arrive in the primal basis (PrismML applies the inverse
 * rotation to table lookups only), so they are rotated forward here and come
 * out of the shared unrotate unchanged.  Positions follow the Qwen3.8
 * multimodal rope (text tokens: t = h = w). */
static bool bonsai2_stage_inputs(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                 const int *tokens, uint32_t T) {
    const uint32_t E = DS4_N_EMBD;
    const ds4_vision_span *spans = g->vis_spans;
    size_t n_spans = g->vis_span_count;
    if (!spans) spans = qwen4_fake_spans(&n_spans);
    for (uint32_t t = 0; t < T; t++) {
        float *dst = g->host_row + (uint64_t)t * E;
        const float *img = n_spans ? qwen4_span_row(spans, n_spans, g->pos + t) : NULL;
        if (img) bonsai2_rotate_row_host(dst, img);
        else bonsai2_pq2_row(m, w->token_embd, (uint64_t)tokens[t], dst);
        qwen4_mrope_pos(spans, n_spans, g->pos + t, &g->mrope_delta, g->host_pos3 + (uint64_t)t * 4u);
        g->host_pos3[(uint64_t)t * 4u + 3u] = 0;
    }
    return ds4_gpu_tensor_write(g->R, 0, g->host_row, (uint64_t)T * E * sizeof(float)) &&
           ds4_gpu_tensor_write(g->pos3, (uint64_t)g->pos * 16u, g->host_pos3, (uint64_t)T * 16u);
}

/* GDN alpha/beta (BF16, read the unrotated normed state): one paired
 * dispatch while decoding, the shared Qwen3.8 projections for prefill. */
static bool bonsai2_gates(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                          uint32_t T) {
    if (T <= 8u && l->lin_alpha->type == DS4_TENSOR_BF16 && l->lin_beta->type == DS4_TENSOR_BF16 &&
        ds4_gpu_bonsai2_gemv_bf16_pair_tensor(g->ga, g->gb, g->xn, m->map, m->size,
                                              l->lin_alpha->abs_offset, l->lin_beta->abs_offset,
                                              (uint32_t)l->lin_alpha->dim[0],
                                              (uint32_t)l->lin_alpha->dim[1], T)) {
        return true;
    }
    return qwen4_gemv(g->ga, m, l->lin_alpha, g->xn, T) &&
           qwen4_gemv(g->gb, m, l->lin_beta, g->xn, T);
}

/* Gated DeltaNet: the Qwen3.8 conv/prep/scan kernels on Bonsai's tiled V-head
 * order.  The output norm is Bonsai's own: Qwen3.5 gates with silu(z) where
 * Qwen3.8 uses sigmoid(z), and ssm_out kept HF grouped order, so the fused
 * kernel also permutes before rotating. */
static bool bonsai2_graph_linear(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                 uint32_t il, uint32_t T) {
    const uint32_t Hk = DS4_N_LIN_K_HEAD, Hv = DS4_N_LIN_V_HEAD, D = DS4_N_LIN_HEAD_DIM;
    return bonsai2_mm(g->qkv, m, l->lin_qkv, g->mixed, T) &&
           bonsai2_mm(g->z, m, l->lin_gate, g->mixed, T) &&
           bonsai2_gates(g, m, l, T) &&
           ds4_gpu_qwen4_conv_stream_tensor(g->qkv, g->layer_lin_hist[il], m->map, m->size,
                                            l->lin_conv->abs_offset, T, DS4_N_LIN_CONV_DIM,
                                            DS4_N_LIN_CONV, true) &&
           ds4_gpu_qwen4_gdn_prep_tensor(g->qkv, g->ga, g->gb, m->map, m->size, l->lin_a->abs_offset,
                                         l->lin_dt_bias->abs_offset, T, Hk, Hv, D) &&
           ds4_gpu_qwen4_gdn_scan_tensor(g->lin_o, g->layer_lin_state[il], g->qkv, g->ga, g->gb,
                                         T, Hk, Hv, D, NULL, 0u, NULL, 1u) &&
           ds4_gpu_bonsai2_gdn_out_rotate_tensor(g->q, g->lin_o, g->z, m->map, m->size,
                                                 l->lin_norm->abs_offset,
                                                 g_bonsai2_sign_gpu[BONSAI2_SIGN_INNER], T, D, Hk,
                                                 Hv / Hk, DS4_RMS_EPS) &&
           bonsai2_mm(g->blk, m, l->lin_out, g->q, T);
}

/* Dense gated GQA over the f16 caches: q/k norms and partial rope in the
 * prep, sigmoid(gate) applied by the Qwen3.8 attention kernels. */
static bool bonsai2_graph_attention(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                    uint32_t il, uint32_t pos0, uint32_t T) {
    const uint32_t H = DS4_N_HEAD, Hkv = DS4_N_HEAD_KV, D = DS4_N_HEAD_DIM;
    return bonsai2_mm(g->qg, m, l->attn_q, g->mixed, T) &&
           bonsai2_mm(g->kp, m, l->attn_k, g->mixed, T) &&
           bonsai2_mm(g->vp, m, l->attn_v, g->mixed, T) &&
           ds4_gpu_bonsai2_attn_prep_tensor(g->q, g->gate, g->layer_k_cache[il], g->layer_v_cache[il],
                                            g->qg, g->kp, g->vp, g->pos3, m->map, m->size,
                                            l->attn_q_norm->abs_offset, l->attn_k_norm->abs_offset,
                                            T, H, Hkv, D, DS4_N_ROT, pos0, g->ctx_cap,
                                            g_qwen4_rope_freq, g_qwen4_rope_mscale, DS4_RMS_EPS) &&
           ds4_gpu_qwen4_attn_decode_tensor(g->attn_o, g->q, g->gate, g->layer_k_cache[il],
                                            g->layer_v_cache[il], NULL, NULL, T <= 2u ? g->attn_part : NULL,
                                            T, H, Hkv, D, pos0, false, 0u, 1.0f / sqrtf((float)D)) &&
           ds4_gpu_bonsai2_rotate_tensor(g->q, g->attn_o, g_bonsai2_sign_gpu[BONSAI2_SIGN_INNER], T,
                                         H * D, 0u, 0u, 0u) &&
           bonsai2_mm(g->blk, m, l->attn_output, g->q, T);
}

static bool bonsai2_graph_ffn(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                              uint32_t T) {
    return bonsai2_norm(g, m, l->ffn_norm, g->R, NULL, T) &&
           bonsai2_mm(g->sh_gate, m, l->ffn_gate, g->mixed, T) &&
           bonsai2_mm(g->sh_up, m, l->ffn_up, g->mixed, T) &&
           ds4_gpu_bonsai2_swiglu_rotate_tensor(g->sh_mid, g->sh_gate, g->sh_up,
                                                g_bonsai2_sign_gpu[BONSAI2_SIGN_FF], T, DS4_N_FF_DENSE) &&
           bonsai2_mm(g->blk, m, l->ffn_down, g->sh_mid, T);
}

/* Command buffer split points.  A decode layer runs ~7x longer on the GPU
 * than it takes to encode, so the GPU starts on two layers, and each later
 * segment is committed before encoding the next one could outrun it. */
static bool bonsai2_flush_after(uint32_t il) {
    return il == 1u || il == 9u || il == 25u;
}

/* Same contract as qwen4_graph_forward_tokens.  Bonsai has no MTP, so only
 * the last row's logits are ever requested. */
static bool bonsai2_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                         const int *tokens, uint32_t T, float *logits_out, bool all_rows) {
    if (!g || T == 0 || T > g->cap_tokens || g->pos + T > g->ctx_cap) return false;
    if (all_rows && T > g->n_logit_rows) return false;
    for (uint32_t t = 0; t < T; t++) {
        if (tokens[t] < 0 || tokens[t] >= (int)DS4_N_VOCAB) {
            fprintf(stderr, "ds4: Bonsai token id %d is outside the vocabulary\n", tokens[t]);
            return false;
        }
    }
    const uint32_t pos0 = g->pos, E = DS4_N_EMBD;
    if (!bonsai2_stage_inputs(g, m, w, tokens, T) || !glm_graph_begin_commands_if_needed()) return false;
    bool ok = ds4_gpu_bonsai2_unrotate_tensor(g->R, g_bonsai2_sign_gpu[BONSAI2_SIGN_EMBD], T, E) != 0;
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        ok = bonsai2_norm(g, m, l->attn_norm, g->R, g->xn, T) &&
             (ds4_qwen4_layer_is_linear(il) ? bonsai2_graph_linear(g, m, l, il, T)
                                            : bonsai2_graph_attention(g, m, l, il, pos0, T)) &&
             qwen4_graph_apply_steering_attn(g, il, T) &&
             ds4_gpu_add_tensor(g->R, g->R, g->blk, T * E) &&
             bonsai2_graph_ffn(g, m, l, T) &&
             ds4_gpu_add_tensor(g->R, g->R, g->blk, T * E) &&
             qwen4_graph_apply_steering_ffn(g, il, T);
        if (ok && bonsai2_flush_after(il)) ok = ds4_gpu_flush_commands() != 0;
    }
    if (ok && logits_out) {
        ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->R, (uint64_t)(T - 1u) * E * sizeof(float),
                                                   (uint64_t)E * sizeof(float));
        ok = last && bonsai2_norm(g, m, w->output_norm, last, NULL, 1) &&
             bonsai2_mm(g->logits, m, w->output, g->mixed, 1);
        ds4_gpu_tensor_free(last);
    }
    if (!ds4_gpu_end_commands()) ok = false;
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits_out, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok) g->pos += T;
    return ok;
}

#else /* CUDA builds share the Qwen3.8 graph code but have no Bonsai kernels */

static bool bonsai2_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens,
                                const ds4_qwen4_gpu_graph *shared,
                                ds4_gpu_tensor *const *state_pool, ds4_gpu_tensor *const *hist_pool,
                                uint32_t slot) {
    (void)ctx_cap; (void)cap_tokens; (void)shared; (void)state_pool; (void)hist_pool; (void)slot;
    memset(g, 0, sizeof(*g));
    fprintf(stderr, "ds4: Ternary Bonsai 2 needs the Metal backend\n");
    return false;
}

static bool bonsai2_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                         const int *tokens, uint32_t T, float *logits_out, bool all_rows) {
    (void)g; (void)m; (void)w; (void)tokens; (void)T; (void)logits_out; (void)all_rows;
    return false;
}

#endif
#endif
