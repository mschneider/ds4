/* Ternary Bonsai 2 batched decode (fork-owned), textually included into
 * ds4.c right before qwen4_graph_encode_native_session_batch, which hands
 * Bonsai batches here.  Same contract as that function: one token per
 * session, rows in item order, the arena's batch_logits receive row i for
 * items[i]; the caller reads them and advances each session's pos.
 *
 * Every PQ2_0 projection and the head run once over all rows (the matvec
 * decodes each weight once for up to eight rows).  The recurrent and
 * attention steps walk each session's own GDN state and KV cache through
 * one-row views with the single-session kernels; for decode they are a
 * small part of the step. */

typedef struct {
    ds4_gpu_tensor *qkv, *ga, *gb, *lin_o, *qg, *kp, *vp, *q, *gate, *attn_o;
} bonsai2_batch_row;

static void bonsai2_batch_rows_free(bonsai2_batch_row *rows, int count) {
    for (int i = 0; i < count; i++) {
        ds4_gpu_tensor **v = (ds4_gpu_tensor **)&rows[i];
        for (size_t k = 0; k < sizeof(rows[i]) / sizeof(*v); k++) ds4_gpu_tensor_free(v[k]);
    }
}

static bool bonsai2_batch_rows_build(bonsai2_batch_row *rows, int count, const ds4_qwen4_gpu_graph *g) {
    const uint64_t v_dim = (uint64_t)DS4_N_LIN_V_HEAD * DS4_N_LIN_HEAD_DIM;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM;
    memset(rows, 0, (size_t)count * sizeof(*rows));
    for (int i = 0; i < count; i++) {
        bonsai2_batch_row *r = &rows[i];
        r->qkv = qwen4_row_view(g->qkv, (uint32_t)i, DS4_N_LIN_CONV_DIM);
        r->ga = qwen4_row_view(g->ga, (uint32_t)i, DS4_N_LIN_V_HEAD);
        r->gb = qwen4_row_view(g->gb, (uint32_t)i, DS4_N_LIN_V_HEAD);
        r->lin_o = qwen4_row_view(g->lin_o, (uint32_t)i, v_dim);
        r->qg = qwen4_row_view(g->qg, (uint32_t)i, 2u * q_dim);
        r->kp = qwen4_row_view(g->kp, (uint32_t)i, kv_dim);
        r->vp = qwen4_row_view(g->vp, (uint32_t)i, kv_dim);
        r->q = qwen4_row_view(g->q, (uint32_t)i, q_dim);
        r->gate = qwen4_row_view(g->gate, (uint32_t)i, q_dim);
        r->attn_o = qwen4_row_view(g->attn_o, (uint32_t)i, q_dim);
        if (!r->qkv || !r->ga || !r->gb || !r->lin_o || !r->qg || !r->kp || !r->vp || !r->q ||
            !r->gate || !r->attn_o) {
            bonsai2_batch_rows_free(rows, i + 1);
            return false;
        }
    }
    return true;
}

static bool bonsai2_graph_encode_session_batch(ds4_decode_item *items, int count, ds4_qwen4_gpu_graph *g,
                                               const ds4_model *m, const ds4_weights *w) {
    const uint32_t T = (uint32_t)count, E = DS4_N_EMBD;
    const uint32_t Hk = DS4_N_LIN_K_HEAD, Hv = DS4_N_LIN_V_HEAD, Dl = DS4_N_LIN_HEAD_DIM;
    const uint32_t H = DS4_N_HEAD, Hkv = DS4_N_HEAD_KV, D = DS4_N_HEAD_DIM;
    if (!qwen4_batch_scratch_ensure(g, T)) return false;

    /* inputs: latent embedding rows (unrotated below) and each session's
     * rope position in its own pos3 table */
    for (int i = 0; i < count; i++) {
        ds4_qwen4_gpu_graph *rg = &items[i].session->qwen4_graph;
        if (items[i].token < 0 || items[i].token >= (int)DS4_N_VOCAB || rg->pos >= rg->ctx_cap) return false;
        bonsai2_pq2_row(m, w->token_embd, (uint64_t)items[i].token, g->host_row + (uint64_t)i * E);
        uint32_t pos3[4];
        qwen4_mrope_pos(NULL, 0, rg->pos, &rg->mrope_delta, pos3);
        pos3[3] = 0;
        if (!ds4_gpu_tensor_write(rg->pos3, (uint64_t)rg->pos * 16u, pos3, 16u)) return false;
    }
    if (!ds4_gpu_tensor_write(g->R, 0, g->host_row, (uint64_t)T * E * sizeof(float))) return false;

    bonsai2_batch_row rows[QWEN4_BATCH_MAX_ROWS];
    if (!bonsai2_batch_rows_build(rows, count, g)) return false;
    bool ok = ds4_gpu_bonsai2_unrotate_tensor(g->R, g_bonsai2_sign_gpu[BONSAI2_SIGN_EMBD], T, E) != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        ok = bonsai2_norm(g, m, l->attn_norm, g->R, g->xn, T);
        if (ok && ds4_qwen4_layer_is_linear(il)) {
            ok = bonsai2_mm(g->qkv, m, l->lin_qkv, g->mixed, T) &&
                 bonsai2_mm(g->z, m, l->lin_gate, g->mixed, T) &&
                 bonsai2_gates(g, m, l, T);
            for (int i = 0; ok && i < count; i++) {
                ok = ds4_gpu_qwen4_conv_stream_tensor(rows[i].qkv, items[i].session->qwen4_graph.layer_lin_hist[il],
                                                      m->map, m->size, l->lin_conv->abs_offset, 1u,
                                                      DS4_N_LIN_CONV_DIM, DS4_N_LIN_CONV, true) != 0;
            }
            ok = ok && ds4_gpu_qwen4_gdn_prep_tensor(g->qkv, g->ga, g->gb, m->map, m->size, l->lin_a->abs_offset,
                                                     l->lin_dt_bias->abs_offset, T, Hk, Hv, Dl);
            for (int i = 0; ok && i < count; i++) {
                ok = ds4_gpu_qwen4_gdn_scan_tensor(rows[i].lin_o, items[i].session->qwen4_graph.layer_lin_state[il],
                                                   rows[i].qkv, rows[i].ga, rows[i].gb, 1u, Hk, Hv, Dl,
                                                   NULL, 0u, NULL, 1u) != 0;
            }
            ok = ok && ds4_gpu_bonsai2_gdn_out_rotate_tensor(g->q, g->lin_o, g->z, m->map, m->size,
                                                             l->lin_norm->abs_offset,
                                                             g_bonsai2_sign_gpu[BONSAI2_SIGN_INNER], T, Dl, Hk,
                                                             Hv / Hk, DS4_RMS_EPS) &&
                 bonsai2_mm(g->blk, m, l->lin_out, g->q, T);
        } else if (ok) {
            ok = bonsai2_mm(g->qg, m, l->attn_q, g->mixed, T) &&
                 bonsai2_mm(g->kp, m, l->attn_k, g->mixed, T) &&
                 bonsai2_mm(g->vp, m, l->attn_v, g->mixed, T);
            for (int i = 0; ok && i < count; i++) {
                ds4_qwen4_gpu_graph *rg = &items[i].session->qwen4_graph;
                ok = ds4_gpu_bonsai2_attn_prep_tensor(rows[i].q, rows[i].gate, rg->layer_k_cache[il],
                                                      rg->layer_v_cache[il], rows[i].qg, rows[i].kp, rows[i].vp,
                                                      rg->pos3, m->map, m->size, l->attn_q_norm->abs_offset,
                                                      l->attn_k_norm->abs_offset, 1u, H, Hkv, D, DS4_N_ROT,
                                                      rg->pos, rg->ctx_cap, g_qwen4_rope_freq,
                                                      g_qwen4_rope_mscale, DS4_RMS_EPS) &&
                     ds4_gpu_qwen4_attn_decode_tensor(rows[i].attn_o, rows[i].q, rows[i].gate,
                                                      rg->layer_k_cache[il], rg->layer_v_cache[il], NULL, NULL,
                                                      g->attn_part, 1u, H, Hkv, D, rg->pos, false, 0u,
                                                      1.0f / sqrtf((float)D));
            }
            ok = ok && ds4_gpu_bonsai2_rotate_tensor(g->q, g->attn_o, g_bonsai2_sign_gpu[BONSAI2_SIGN_INNER], T,
                                                     H * D, 0u, 0u, 0u) &&
                 bonsai2_mm(g->blk, m, l->attn_output, g->q, T);
        }
        ok = ok && ds4_gpu_add_tensor(g->R, g->R, g->blk, T * E) &&
             bonsai2_graph_ffn(g, m, l, T) &&
             ds4_gpu_add_tensor(g->R, g->R, g->blk, T * E);
        if (ok && bonsai2_flush_after(il)) ok = ds4_gpu_flush_commands() != 0;
    }
    ok = ok && bonsai2_norm(g, m, w->output_norm, g->R, NULL, T) &&
         bonsai2_mm(g->batch_logits, m, w->output, g->mixed, T);
    bonsai2_batch_rows_free(rows, count);
    return ok;
}
