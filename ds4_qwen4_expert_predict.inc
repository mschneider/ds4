/* Next-layer expert prediction for SSD streaming (fork).
 *
 * In a streamed single-token decode step, layer l+1's router runs on layer
 * l's MoE input as well, and its top-k experts start loading from the SSD
 * once layer l's own experts are submitted. The reads then overlap layer l's
 * expert work and layer l+1's mixer instead of waiting behind layer l+1's
 * router. The real routing is unchanged: a wrong guess only costs a read and
 * a cache slot. ds4_metal_expert_predict.inc runs the loads.
 *
 * Off by default: on an M4 Pro 48 GB with the Q2 pack it did not speed up
 * decode (docs/QWEN38_FLASH_NEXT.md). DS4_QWEN4_STREAM_PREDICT=1 turns it on.
 *
 * Included by ds4.c before qwen4_graph_moe_impl. */

int ds4_gpu_qwen4_stream_predict_set(const ds4_gpu_stream_expert_table *next,
                                     const ds4_gpu_tensor *ids, uint32_t n_ids);

#ifdef DS4_HAS_QWEN4_METAL
static bool qwen4_stream_predict_next(ds4_qwen4_gpu_graph *g, const ds4_model *m,
                                      const ds4_layer_weights *l, uint32_t il, uint32_t T) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = getenv("DS4_QWEN4_STREAM_PREDICT");
        enabled = env && strcmp(env, "1") == 0;
    }
    /* Callers pass l = &w->layer[il]; the trunk's last layer has no successor
     * and the MTP layer is not part of the trunk. */
    if (!enabled || !g->ssd_streaming || T != 1u || il + 1u >= DS4_N_LAYER - DS4_N_NEXTN_PREDICT) return true;
    const ds4_layer_weights *next = l + 1;
    uint64_t gate_bytes = 0, down_bytes = 0;
    if (!streaming_layer_gate_down_expert_bytes(next, &gate_bytes, &down_bytes) ||
        next->ffn_gate_exps->type != next->ffn_up_exps->type) return true;
    const ds4_gpu_tensor *router = qwen4_bf16_weights(m, next->ffn_gate_inp);
    if (!router) return true;

    static ds4_gpu_tensor *logits, *ids, *weights, *shared;
    if (!logits) {
        logits = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT * sizeof(float));
        ids = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
        weights = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EXPERT_USED * sizeof(float));
        shared = ds4_gpu_tensor_alloc(sizeof(float));
        if (!logits || !ids || !weights || !shared) {
            enabled = 0;
            return true;
        }
    }
    if (!ds4_gpu_matmul_bf16_weights_tensor(logits, router, (uint32_t)next->ffn_gate_inp->dim[0],
                                            (uint32_t)next->ffn_gate_inp->dim[1], g->mixed) ||
        !ds4_gpu_qwen4_router_topk_tensor(ids, weights, logits, g->mixed, m->map, m->size,
                                          next->ffn_gate_inp_shexp->abs_offset, next->ffn_gate_inp_shexp->type,
                                          DS4_N_EMBD, shared, 1, DS4_N_EXPERT, DS4_N_EXPERT_USED)) return false;
    const ds4_gpu_stream_expert_table table = {
        .model_map = m->map, .model_size = m->size, .layer = il + 1u,
        .n_total_expert = DS4_N_EXPERT,
        .gate_offset = next->ffn_gate_exps->abs_offset,
        .up_offset = next->ffn_up_exps->abs_offset,
        .down_offset = next->ffn_down_exps->abs_offset,
        .gate_expert_bytes = gate_bytes, .down_expert_bytes = down_bytes,
    };
    return ds4_gpu_qwen4_stream_predict_set(&table, ids, DS4_N_EXPERT_USED) != 0;
}
#endif
