/* Dense qwen35 GPU graph (fork-owned): Ternary Bonsai 2 and the plain
 * Qwen3.8-27B.  Textually included into ds4.c right after the Qwen3.8 GPU
 * graph section, so the Qwen3.8 graph type, its GDN/attention kernel wrappers
 * and helpers are visible here.
 *
 * A session of either variant is an ordinary Qwen3.8 session whose graph
 * allocation and forward pass dispatch here (qwen4_graph_alloc /
 * qwen4_graph_forward_tokens), so recurrent state, KV caches, positions,
 * snapshots and disk payloads use the Qwen3.8 fields and code unchanged.  The
 * graph itself is the dense Qwen3.8-27B decoder:
 *
 *   h = embd[token]
 *   per layer:  x = rms(h) * attn_norm
 *               h += mixer(x)                          GDN or gated attention
 *               x = rms(h) * post_attention_norm
 *               h += down(silu(gate(x)) * up(x))
 *   logits = output(rms(h) * output_norm)
 *
 * Bonsai stores every projection as PQ2_0 with a folded Hadamard rotation, so
 * its inputs are rotated once (sign, then blockwise FWHT) and shared by all
 * projections that read them: qkv/z or q/k/v off the normed hidden state,
 * gate/up off the FFN input; its embedding holds latent rows, unrotated after
 * staging.  The plain model skips every transform; the helpers below are the
 * only places the two differ.  Buffer roles: R residual, xn unrotated normed
 * (Bonsai's BF16 GDN gates), mixed normed (rotated for Bonsai), blk mixer/FFN
 * output, q Bonsai's rotated 6144-wide input of ssm_out and attn_output,
 * sh_gate/sh_up/sh_mid the FFN. */

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


/* A projection, by weight type: PQ2_0 (Bonsai), Q4_K, Q6_K, or the Qwen3.8
 * dense types (Q8_0, Q4_0, BF16, F16, F32).  rows (0: all) keeps a leading prefix
 * of the output rows, for the MTP draft head.  Q4_K takes the classic
 * matvec for one or two rows (one pass each, so a two-row verify row is
 * bit-identical to a decode step) and the simdgroup-matrix product for
 * 3-8, where it is cheaper. */
static bool bonsai2_mm_rows(ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w,
                            const ds4_gpu_tensor *x, uint32_t T, uint32_t rows) {
    if (!rows || rows > w->dim[1]) rows = (uint32_t)w->dim[1];
    const uint32_t in = (uint32_t)w->dim[0];
    if (w->type == DS4_TENSOR_Q6_K) {
        if (ds4_gpu_bonsai2_matmul_q6k_tensor(out, m->map, m->size, w->abs_offset, in, rows, x, T)) return true;
        fprintf(stderr, "ds4: qwen35 Q6_K matmul failed for %.*s (%u tokens)\n", (int)w->name.len, w->name.ptr, T);
        return false;
    }
    if (w->type != DS4_TENSOR_PQ2_0 && w->type != DS4_TENSOR_Q4_K) return qwen4_gemv_rows(out, m, w, x, T, rows);
    const int ok = w->type == DS4_TENSOR_PQ2_0
        ? ds4_gpu_bonsai2_matmul_pq2_tensor(out, m->map, m->size, w->abs_offset, in, rows, x, T)
        : (T >= 3u && T <= 8u &&
           ds4_gpu_bonsai2_mm_q4k_rows_tensor(out, m->map, m->size, w->abs_offset, in, rows, x, T)) ||
          ds4_gpu_matmul_quant_tensor(out, m->map, m->size, w->abs_offset, w->type, in, rows, x, T);
    if (ok) return true;
    fprintf(stderr, "ds4: qwen35 matmul failed for %.*s (type %u, %u tokens)\n",
            (int)w->name.len, w->name.ptr, w->type, T);
    return false;
}

static bool bonsai2_mm(ds4_gpu_tensor *out, const ds4_model *m, const ds4_tensor *w,
                       const ds4_gpu_tensor *x, uint32_t T) {
    return bonsai2_mm_rows(out, m, w, x, T, 0u);
}

/* Normed hidden state into g->mixed.  Bonsai rotates it for the PQ2_0
 * projections, and xn (optional) keeps the unrotated copy for its BF16 GDN
 * gates; the plain model's gates read g->mixed directly. */
static bool bonsai2_norm(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_tensor *norm,
                         const ds4_gpu_tensor *x, ds4_gpu_tensor *xn, uint32_t T) {
    if (!ds4_model_is_bonsai2()) {
        return ds4_gpu_rms_norm_weight_rows_tensor(g->mixed, x, m->map, m->size, norm->abs_offset,
                                                   DS4_N_EMBD, T, DS4_RMS_EPS) != 0;
    }
    return ds4_gpu_bonsai2_rms_rotate_tensor(g->mixed, xn, x, m->map, m->size, norm->abs_offset,
                                             g_bonsai2_sign_gpu[BONSAI2_SIGN_EMBD], T, DS4_N_EMBD,
                                             DS4_RMS_EPS) != 0;
}

/* GDN output norm and gate; returns the input of ssm_out.  Bonsai also
 * permutes and rotates it into g->q; the plain model gates g->lin_o in
 * place. */
static ds4_gpu_tensor *bonsai2_gdn_out(ds4_qwen4_gpu_graph *g, const ds4_model *m,
                                       const ds4_layer_weights *l, uint32_t T) {
    const uint32_t Hk = DS4_N_LIN_K_HEAD, Hv = DS4_N_LIN_V_HEAD, D = DS4_N_LIN_HEAD_DIM;
    if (!ds4_model_is_bonsai2()) {
        return ds4_gpu_bonsai2_gdn_out_silu_tensor(g->lin_o, g->z, m->map, m->size,
                                                   l->lin_norm->abs_offset, T, Hv, DS4_RMS_EPS)
            ? g->lin_o : NULL;
    }
    return ds4_gpu_bonsai2_gdn_out_rotate_tensor(g->q, g->lin_o, g->z, m->map, m->size,
                                                 l->lin_norm->abs_offset,
                                                 g_bonsai2_sign_gpu[BONSAI2_SIGN_INNER], T, D, Hk,
                                                 Hv / Hk, DS4_RMS_EPS)
        ? g->q : NULL;
}

/* Input of attn_output: the gated attention rows, rotated into g->q for
 * Bonsai. */
static ds4_gpu_tensor *bonsai2_attn_out(ds4_qwen4_gpu_graph *g, uint32_t T) {
    if (!ds4_model_is_bonsai2()) return g->attn_o;
    return ds4_gpu_bonsai2_rotate_tensor(g->q, g->attn_o, g_bonsai2_sign_gpu[BONSAI2_SIGN_INNER], T,
                                         DS4_N_HEAD * DS4_N_HEAD_DIM, 0u, 0u, 0u)
        ? g->q : NULL;
}

/* Bonsai's staged rows are latent: h = s * H(z).  Plain rows are final. */
static bool bonsai2_embed_finish(ds4_qwen4_gpu_graph *g, uint32_t T) {
    return !ds4_model_is_bonsai2() ||
           ds4_gpu_bonsai2_unrotate_tensor(g->R, g_bonsai2_sign_gpu[BONSAI2_SIGN_EMBD], T, DS4_N_EMBD);
}

static ds4_gpu_tensor *bonsai2_alloc_f32(uint64_t n) {
    return qwen4_graph_alloc_f32(n ? n : 1u);
}

/* Rows per MTP pass: drafts and verify catch-ups use 1-3, prefill fills the
 * MTP layer's KV cache in slices of this many. */
#define BONSAI2_MTP_ROWS 64u

static bool bonsai2_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens, bool mtp,
                                const ds4_qwen4_gpu_graph *shared,
                                ds4_gpu_tensor *const *state_pool, ds4_gpu_tensor *const *hist_pool,
                                uint32_t slot) {
    memset(g, 0, sizeof(*g));
    mtp = mtp && DS4_N_NEXTN_PREDICT != 0 && cap_tokens >= 3u;
    if (ds4_model_is_bonsai2() && !bonsai2_signs_ready()) {
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
    g->n_logit_rows = mtp ? 3u : 1u;
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
    BONSAI2_ALLOC(logits, (uint64_t)g->n_logit_rows * DS4_N_VOCAB);
    /* MTP: the predictor's inputs and residual for up to three rows, and the
     * recurrent snapshots the speculative cycle swaps in on a rejection
     * (after verify row 0, after row 1, and before the block for exact
     * sampling).  Flash Next sizes the snapshot sets lazily from its PLE
     * history, which is empty here, so all three are allocated now: 3 x 151
     * MB of GDN state and conv history. */
    if (mtp) {
        BONSAI2_ALLOC(mtp_e, BONSAI2_MTP_ROWS * E);
        BONSAI2_ALLOC(mtp_cat, BONSAI2_MTP_ROWS * 2u * E);
        BONSAI2_ALLOC(mtp_R, BONSAI2_MTP_ROWS * E);
        BONSAI2_ALLOC(mtp_argmax, 1);
        BONSAI2_ALLOC(mtp_argmax_tmp, ((DS4_N_VOCAB + 4095u) / 4096u) * 2u);
        BONSAI2_ALLOC(snap_ple_hist, 1);
        BONSAI2_ALLOC(snap2_ple_hist, 1);
        BONSAI2_ALLOC(snap0_ple_hist, 1);
    }
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
            if (ok && mtp) {
                BONSAI2_ALLOC(snap_lin_state[il], v_dim * DS4_N_LIN_HEAD_DIM);
                BONSAI2_ALLOC(snap_lin_hist[il], (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim);
                BONSAI2_ALLOC(snap2_lin_state[il], v_dim * DS4_N_LIN_HEAD_DIM);
                BONSAI2_ALLOC(snap2_lin_hist[il], (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim);
                BONSAI2_ALLOC(snap0_lin_state[il], v_dim * DS4_N_LIN_HEAD_DIM);
                BONSAI2_ALLOC(snap0_lin_hist[il], (uint64_t)(DS4_N_LIN_CONV - 1u) * conv_dim);
            }
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
    g->host_logits = xmalloc((uint64_t)g->n_logit_rows * DS4_N_VOCAB * sizeof(float));
    if (!ok) {
        fprintf(stderr, "ds4: %s graph allocation failed (ctx %u)\n", g_ds4_shape.name, ctx_cap);
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

/* Defined with the Qwen3.8 reference code later in ds4.c: host
 * dequantization of one row of any supported weight type. */
static void qwen4_ref_row(const ds4_model *m, const ds4_tensor *t, uint64_t row, float *out);

/* Token rows and rope positions for T tokens at g->pos.  Bonsai's text rows
 * are latent (rotated) PQ2_0 embedding rows, un-rotated on the GPU for the
 * whole batch (bonsai2_embed_finish).  Image rows arrive in the primal basis
 * (PrismML applies the inverse rotation to table lookups only), so for Bonsai
 * they are rotated forward here and come out of the shared unrotate
 * unchanged.  Positions follow the Qwen3.8 multimodal rope (text tokens:
 * t = h = w). */
static bool bonsai2_stage_inputs(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                 const int *tokens, uint32_t T) {
    const uint32_t E = DS4_N_EMBD;
    const ds4_vision_span *spans = g->vis_spans;
    size_t n_spans = g->vis_span_count;
    if (!spans) spans = qwen4_fake_spans(&n_spans);
    for (uint32_t t = 0; t < T; t++) {
        float *dst = g->host_row + (uint64_t)t * E;
        const float *img = n_spans ? qwen4_span_row(spans, n_spans, g->pos + t) : NULL;
        if (img && ds4_model_is_bonsai2()) bonsai2_rotate_row_host(dst, img);
        else if (img) memcpy(dst, img, (uint64_t)E * sizeof(float));
        else qwen4_ref_row(m, w->token_embd, (uint64_t)tokens[t], dst);
        qwen4_mrope_pos(spans, n_spans, g->pos + t, &g->mrope_delta, g->host_pos3 + (uint64_t)t * 4u);
        g->host_pos3[(uint64_t)t * 4u + 3u] = 0;
        /* what a snapshot restore puts back besides the recurrent state */
        if (t == 0 && g->snap_after_first) {
            g->snap_mrope_delta = g->mrope_delta;
            g->snap_pos = g->pos + 1u;
        }
        if (t == 1u && g->snap_after_second) {
            g->snap2_mrope_delta = g->mrope_delta;
            g->snap2_pos = g->pos + 2u;
        }
    }
    return ds4_gpu_tensor_write(g->R, 0, g->host_row, (uint64_t)T * E * sizeof(float)) &&
           ds4_gpu_tensor_write(g->pos3, (uint64_t)g->pos * 16u, g->host_pos3, (uint64_t)T * 16u);
}

/* GDN alpha/beta (read the unrotated normed state): one paired dispatch
 * while decoding BF16 gates, the shared projections otherwise. */
static bool bonsai2_gates(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                          uint32_t T) {
    const ds4_gpu_tensor *x = ds4_model_is_bonsai2() ? g->xn : g->mixed;
    if (T <= 8u && l->lin_alpha->type == DS4_TENSOR_BF16 && l->lin_beta->type == DS4_TENSOR_BF16 &&
        ds4_gpu_bonsai2_gemv_bf16_pair_tensor(g->ga, g->gb, x, m->map, m->size,
                                              l->lin_alpha->abs_offset, l->lin_beta->abs_offset,
                                              (uint32_t)l->lin_alpha->dim[0],
                                              (uint32_t)l->lin_alpha->dim[1], T)) {
        return true;
    }
    return bonsai2_mm(g->ga, m, l->lin_alpha, x, T) &&
           bonsai2_mm(g->gb, m, l->lin_beta, x, T);
}

/* Gated DeltaNet: the Qwen3.8 conv/prep/scan kernels on the tiled V-head
 * order.  The output norm is this family's own: Qwen3.8-27B gates with
 * silu(z) where Qwen3.8 Flash Next uses sigmoid(z); Bonsai's ssm_out also
 * kept HF grouped order, so its fused kernel permutes before rotating. */
static bool bonsai2_graph_linear(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                 uint32_t il, uint32_t T) {
    const uint32_t Hk = DS4_N_LIN_K_HEAD, Hv = DS4_N_LIN_V_HEAD, D = DS4_N_LIN_HEAD_DIM;
    /* During an MTP verify the conv history and the scan state are saved
     * right after rows 0 and 1, so a rejected draft only swaps them in. */
    ds4_gpu_tensor *snap_hist = g->snap_after_first ? g->snap_lin_hist[il] : NULL;
    ds4_gpu_tensor *snap2_hist = g->snap_after_second ? g->snap2_lin_hist[il] : NULL;
    ds4_gpu_tensor *snap_state = g->snap_after_first ? g->snap_lin_state[il] : NULL;
    ds4_gpu_tensor *snap2_state = g->snap_after_second ? g->snap2_lin_state[il] : NULL;
    ds4_gpu_tensor *out_in = NULL;
    if (!bonsai2_mm(g->qkv, m, l->lin_qkv, g->mixed, T) || !bonsai2_mm(g->z, m, l->lin_gate, g->mixed, T)) {
        return false;
    }
    /* Plain decode and verify steps (1-3 rows) take Flash Next's fused
     * front: alpha/beta, conv with history snapshots and prep in one
     * dispatch, the same arithmetic for every row count.  Bonsai's gates read
     * the unrotated xn, which the front does not take. */
    bool front;
    if (!ds4_model_is_bonsai2() && T <= 3u && l->lin_alpha->type == l->lin_beta->type) {
        front = ds4_gpu_qwen4_gdn_front_tensor(g->qkv, g->layer_lin_hist[il], g->mixed, g->ga, g->gb,
                                               m->map, m->size, l->lin_conv->abs_offset,
                                               l->lin_alpha->abs_offset, l->lin_beta->abs_offset,
                                               l->lin_a->abs_offset, l->lin_dt_bias->abs_offset,
                                               l->lin_alpha->type, T, Hk, Hv, D, DS4_N_LIN_CONV, DS4_N_EMBD,
                                               snap_hist, 0u, snap2_hist, 1u) != 0;
    } else if (snap_hist || snap2_hist) {
        return false;   /* only the fused front saves conv history */
    } else {
        front = bonsai2_gates(g, m, l, T) &&
                ds4_gpu_qwen4_conv_stream_tensor(g->qkv, g->layer_lin_hist[il], m->map, m->size,
                                                 l->lin_conv->abs_offset, T, DS4_N_LIN_CONV_DIM,
                                                 DS4_N_LIN_CONV, true) &&
                ds4_gpu_qwen4_gdn_prep_tensor(g->qkv, g->ga, g->gb, m->map, m->size, l->lin_a->abs_offset,
                                              l->lin_dt_bias->abs_offset, T, Hk, Hv, D);
    }
    return front &&
           ds4_gpu_qwen4_gdn_scan_tensor(g->lin_o, g->layer_lin_state[il], g->qkv, g->ga, g->gb,
                                         T, Hk, Hv, D, snap_state, 0u, snap2_state, 1u) &&
           (out_in = bonsai2_gdn_out(g, m, l, T)) &&
           bonsai2_mm(g->blk, m, l->lin_out, out_in, T);
}

/* Dense gated GQA over the f16 caches: q/k norms and partial rope in the
 * prep, sigmoid(gate) applied by the Qwen3.8 attention kernels. */
static bool bonsai2_graph_attention(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                                    uint32_t il, uint32_t pos0, uint32_t T) {
    const uint32_t H = DS4_N_HEAD, Hkv = DS4_N_HEAD_KV, D = DS4_N_HEAD_DIM;
    ds4_gpu_tensor *out_in = NULL;
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
           (out_in = bonsai2_attn_out(g, T)) &&
           bonsai2_mm(g->blk, m, l->attn_output, out_in, T);
}

static bool bonsai2_graph_ffn(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_layer_weights *l,
                              uint32_t T) {
    if (!bonsai2_norm(g, m, l->ffn_norm, g->R, NULL, T) ||
        !bonsai2_mm(g->sh_gate, m, l->ffn_gate, g->mixed, T) ||
        !bonsai2_mm(g->sh_up, m, l->ffn_up, g->mixed, T)) {
        return false;
    }
    const bool mid = ds4_model_is_bonsai2()
        ? ds4_gpu_bonsai2_swiglu_rotate_tensor(g->sh_mid, g->sh_gate, g->sh_up,
                                               g_bonsai2_sign_gpu[BONSAI2_SIGN_FF], T, DS4_N_FF_DENSE) != 0
        : ds4_gpu_swiglu_tensor(g->sh_mid, g->sh_gate, g->sh_up, T * DS4_N_FF_DENSE, 0.0f, 1.0f) != 0;
    return mid && bonsai2_mm(g->blk, m, l->ffn_down, g->sh_mid, T);
}

/* Command buffer split points.  A decode layer runs ~7x longer on the GPU
 * than it takes to encode, so the GPU starts on two layers, and each later
 * segment is committed before encoding the next one could outrun it. */
static bool bonsai2_flush_after(uint32_t il) {
    return il == 1u || il == 9u || il == 25u;
}

static bool bonsai2_mtp_pass(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                             ds4_gpu_tensor *src, uint32_t src_row, const ds4_tensor *src_norm,
                             const int *next_tokens, uint32_t T, uint32_t idx, float *logits_out,
                             int *draft_out);

/* Same contract as qwen4_graph_forward_tokens: the trunk layers only (the
 * MTP layer runs in bonsai2_graph_mtp_steps), logits for the last row or,
 * for MTP verifies, every row.
 *
 * With MTP, a prefill also runs the MTP layer over every row whose next
 * token it already holds, as llama.cpp does, so drafts attend over a filled
 * cache rather than empty prompt rows: on the code reference prompt this
 * lifts acceptance from 67% to 81%.  The last row of a chunk waits for its
 * token; at a chunk boundary that one row stays unfilled. */
static bool bonsai2_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                         const int *tokens, uint32_t T, float *logits_out, bool all_rows) {
    if (!g || T == 0 || T > g->cap_tokens || g->pos + T > g->ctx_cap) return false;
    if (all_rows && T > g->n_logit_rows) return false;
    for (uint32_t t = 0; t < T; t++) {
        if (tokens[t] < 0 || tokens[t] >= (int)DS4_N_VOCAB) {
            fprintf(stderr, "ds4: qwen35 token id %d is outside the vocabulary\n", tokens[t]);
            return false;
        }
    }
    const uint32_t pos0 = g->pos, E = DS4_N_EMBD;
    if (!bonsai2_stage_inputs(g, m, w, tokens, T) || !glm_graph_begin_commands_if_needed()) return false;
    const uint32_t n_trunk = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    const bool verify = g->snap_after_first || g->snap_after_second;
    bool ok = bonsai2_embed_finish(g, T);
    for (uint32_t il = 0; il < n_trunk && ok; il++) {
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
    const uint32_t rows = all_rows ? T : 1u;
    if (ok && logits_out) {
        ds4_gpu_tensor *tail = ds4_gpu_tensor_view(g->R, (uint64_t)(T - rows) * E * sizeof(float),
                                                   (uint64_t)rows * E * sizeof(float));
        ok = tail && bonsai2_norm(g, m, w->output_norm, tail, NULL, rows) &&
             bonsai2_mm(g->logits, m, w->output, g->mixed, rows);
        ds4_gpu_tensor_free(tail);
    }
    for (uint32_t r = 0; ok && g->mtp_R && !verify && r + 1u < T; r += BONSAI2_MTP_ROWS) {
        const uint32_t n = T - 1u - r < BONSAI2_MTP_ROWS ? T - 1u - r : BONSAI2_MTP_ROWS;
        ok = bonsai2_mtp_pass(g, m, w, g->R, r, w->output_norm, tokens + r + 1u, n, pos0 + r, NULL, NULL);
    }
    if (!ds4_gpu_end_commands()) ok = false;
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits_out, (uint64_t)rows * DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok) g->pos += T;
    if (g->snap_after_first) g->snap_valid = ok;
    if (g->snap_after_second) g->snap2_valid = ok;
    g->snap_after_first = g->snap_after_second = false;
    return ok;
}

/* One causal pass of the MTP layer (the last layer, past the trunk) over T
 * rows at its KV index idx, as llama.cpp's qwen35 graph_mtp:
 *
 *   x = eh_proj([enorm(embd(next_tokens[t])), hnorm(src_norm(h[t]))])
 *   x += attention(attn_norm(x))          own KV cache, rope at idx + t
 *   x += ffn(post_attention_norm(x))
 *   draft = argmax(output(shared_head_norm(x[T-1])))
 *
 * h[t] is row src_row + t of src: the trunk residual (src_norm output_norm)
 * when catching up on verified rows, or the predictor's own last row
 * (src_norm shared_head_norm) when chaining a second draft.  The residual
 * stays in mtp_R for the chain.  Positions past the trunk's get text rope
 * positions here; the trunk rewrites them when it gets there.  Without
 * draft_out and logits_out the pass only writes the layer's KV rows (prefill)
 * and leaves the command batch open. */
static bool bonsai2_mtp_pass(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                             ds4_gpu_tensor *src, uint32_t src_row, const ds4_tensor *src_norm,
                             const int *next_tokens, uint32_t T, uint32_t idx, float *logits_out,
                             int *draft_out) {
    const uint32_t E = DS4_N_EMBD, il = DS4_N_LAYER - 1u;
    const ds4_layer_weights *l = &w->layer[il];
    const bool head = logits_out || draft_out;
    if (!g->mtp_R || !src || T == 0u || T > BONSAI2_MTP_ROWS || T > g->cap_tokens || idx > g->ctx_cap ||
        T > g->ctx_cap - idx) {
        return false;
    }
    for (uint32_t t = 0; t < T; t++) {
        if (next_tokens[t] < 0 || next_tokens[t] >= (int)DS4_N_VOCAB) return false;
        qwen4_ref_row(m, w->token_embd, (uint64_t)next_tokens[t], g->host_row + (uint64_t)t * E);
    }
    if (!ds4_gpu_tensor_write(g->mtp_e, 0, g->host_row, (uint64_t)T * E * sizeof(float))) return false;
    int32_t delta = g->mrope_delta;
    for (uint32_t p = idx; p < idx + T; p++) {
        if (p < g->pos) continue;
        uint32_t pos3[4] = {0};
        qwen4_mrope_pos(NULL, 0, p, &delta, pos3);
        if (!ds4_gpu_tensor_write(g->pos3, (uint64_t)p * 16u, pos3, 16u)) return false;
    }
    if (!glm_graph_begin_commands_if_needed()) return false;
    const uint64_t row = (uint64_t)E * sizeof(float);
    bool ok = true;
    for (uint32_t t = 0; t < T && ok; t++) {
        ds4_gpu_tensor *e = ds4_gpu_tensor_view(g->mtp_e, t * row, row);
        ds4_gpu_tensor *h = ds4_gpu_tensor_view(src, (uint64_t)(src_row + t) * row, row);
        ds4_gpu_tensor *hn = ds4_gpu_tensor_view(g->xn, t * row, row);
        ds4_gpu_tensor *cat_e = ds4_gpu_tensor_view(g->mtp_cat, 2u * t * row, row);
        ds4_gpu_tensor *cat_h = ds4_gpu_tensor_view(g->mtp_cat, (2u * t + 1u) * row, row);
        ok = e && h && hn && cat_e && cat_h &&
             ds4_gpu_rms_norm_weight_rows_tensor(cat_e, e, m->map, m->size, l->nextn_enorm->abs_offset,
                                                 E, 1, DS4_RMS_EPS) &&
             ds4_gpu_rms_norm_weight_rows_tensor(hn, h, m->map, m->size, src_norm->abs_offset,
                                                 E, 1, DS4_RMS_EPS) &&
             ds4_gpu_rms_norm_weight_rows_tensor(cat_h, hn, m->map, m->size, l->nextn_hnorm->abs_offset,
                                                 E, 1, DS4_RMS_EPS);
        ds4_gpu_tensor_free(cat_h);
        ds4_gpu_tensor_free(cat_e);
        ds4_gpu_tensor_free(hn);
        ds4_gpu_tensor_free(h);
        ds4_gpu_tensor_free(e);
    }
    ds4_gpu_tensor *R_save = g->R;
    ok = ok && bonsai2_mm(g->mtp_R, m, l->nextn_eh_proj, g->mtp_cat, T);
    g->R = g->mtp_R;
    ok = ok && bonsai2_norm(g, m, l->attn_norm, g->R, g->xn, T) &&
         bonsai2_graph_attention(g, m, l, il, idx, T) &&
         ds4_gpu_add_tensor(g->R, g->R, g->blk, T * E) &&
         bonsai2_graph_ffn(g, m, l, T) &&
         ds4_gpu_add_tensor(g->R, g->R, g->blk, T * E);
    g->R = R_save;
    /* the draft needs only its argmax; DS4_QWEN4_MTP_DRAFT_ROWS scores the
     * frequent-token prefix, as for Flash Next */
    if (!head) {
        if (ok) {
            g->mtp_pos = idx + T;
            g->mtp_last_rows = 0;   /* no chain from a fill pass */
        }
        return ok;
    }
    const uint32_t head_rows = logits_out ? DS4_N_VOCAB : qwen4_mtp_draft_rows();
    ds4_gpu_tensor *last = ds4_gpu_tensor_view(g->mtp_R, (uint64_t)(T - 1u) * row, row);
    ok = ok && last && bonsai2_norm(g, m, l->nextn_shared_head_norm, last, NULL, 1) &&
         bonsai2_mm_rows(g->logits, m, w->output, g->mixed, 1, head_rows) &&
         (logits_out || ds4_gpu_qwen4_argmax_tensor(g->mtp_argmax, g->mtp_argmax_tmp, g->logits, head_rows));
    ds4_gpu_tensor_free(last);
    if (!ds4_gpu_end_commands()) ok = false;
    if (ok && logits_out) {
        ok = ds4_gpu_tensor_read(g->logits, 0, logits_out, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        if (ok && draft_out) *draft_out = sample_argmax(logits_out, DS4_N_VOCAB);
    } else if (ok && draft_out) {
        int32_t token = -1;
        ok = ds4_gpu_tensor_read(g->mtp_argmax, 0, &token, sizeof(token)) != 0 &&
             token >= 0 && (uint32_t)token < head_rows;
        if (ok) *draft_out = token;
    }
    if (ok) {
        g->mtp_pos = idx + T;
        g->mtp_last_rows = T;
    }
    return ok;
}

/* qwen4_graph_mtp_steps for this family: rows row.. of the last trunk
 * forward, catching up on verified tokens and drafting after the last. */
static bool bonsai2_graph_mtp_steps(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                    uint32_t row, const int *next_tokens, uint32_t T, uint32_t idx,
                                    bool want_logits, float *logits_out, int *draft_out) {
    if (!want_logits || T > 3u) return false;   /* no batched predictor for this family */
    return bonsai2_mtp_pass(g, m, w, g->R, row, w->output_norm, next_tokens, T, idx, logits_out, draft_out);
}

/* qwen4_graph_mtp_chain_step: one more draft from the predictor's own last
 * row, a proposal the next verify checks like any other. */
static bool bonsai2_graph_mtp_chain_step(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                         int next_token, uint32_t idx, int *draft_out) {
    if (g->mtp_last_rows == 0u || g->mtp_last_rows > 3u) return false;
    const ds4_layer_weights *l = &w->layer[DS4_N_LAYER - 1u];
    return bonsai2_mtp_pass(g, m, w, g->mtp_R, g->mtp_last_rows - 1u, l->nextn_shared_head_norm,
                            &next_token, 1u, idx, NULL, draft_out);
}

#else /* CUDA builds share the Qwen3.8 graph code but have no qwen35 kernels */

static bool bonsai2_graph_alloc(ds4_qwen4_gpu_graph *g, uint32_t ctx_cap, uint32_t cap_tokens, bool mtp,
                                const ds4_qwen4_gpu_graph *shared,
                                ds4_gpu_tensor *const *state_pool, ds4_gpu_tensor *const *hist_pool,
                                uint32_t slot) {
    (void)ctx_cap; (void)cap_tokens; (void)mtp; (void)shared; (void)state_pool; (void)hist_pool; (void)slot;
    memset(g, 0, sizeof(*g));
    fprintf(stderr, "ds4: %s needs the Metal backend\n", g_ds4_shape.name);
    return false;
}

static bool bonsai2_graph_forward_tokens(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                         const int *tokens, uint32_t T, float *logits_out, bool all_rows) {
    (void)g; (void)m; (void)w; (void)tokens; (void)T; (void)logits_out; (void)all_rows;
    return false;
}

static bool bonsai2_graph_mtp_steps(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                    uint32_t row, const int *next_tokens, uint32_t T, uint32_t idx,
                                    bool want_logits, float *logits_out, int *draft_out) {
    (void)g; (void)m; (void)w; (void)row; (void)next_tokens; (void)T; (void)idx; (void)want_logits;
    (void)logits_out; (void)draft_out;
    return false;
}

static bool bonsai2_graph_mtp_chain_step(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                         int next_token, uint32_t idx, int *draft_out) {
    (void)g; (void)m; (void)w; (void)next_token; (void)idx; (void)draft_out;
    return false;
}

#endif
#endif
