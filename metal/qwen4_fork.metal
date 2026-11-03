// Fork-owned Qwen3.8 kernels. Appended after qwen4.metal, so its helpers and
// argument structs are in scope.

/* Q8_0 hyper-connection gate/mix for decode: the low-rank SiLU is computed
 * once per threadgroup, as in the F16 reuse kernel. One SIMD group per output
 * row; its four 8-lane groups own the four streams' rows, and lane k reads
 * the k-th four quants of every block, so each group reads its row's 32-byte
 * quant runs in one pass. */
kernel void kernel_qwen4_fork_hc_gate_mix_q8(
        constant ds4_metal_args_qwen4_hc_gate_mix & args,
        device const float *xn,
        device const float *lo,
        device const char  *w_up,
        device float       *mixed,
        threadgroup float4 *act [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint hc = args.n_hc, E = args.n_embd, rank = args.n_rank;
    const uint d = tgpig.x * (ntg.x / 32) + sgitg;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    device const float4 *l = (device const float4 *)(lo + (uint64_t)tok * rank);
    const float inv_hc = 1.0f / (float)hc;
    for (uint r = tid; r < rank / 4u; r += ntg.x) {
        const float4 x = l[r] * inv_hc;
        act[r] = x / (1.0f + exp(-x));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= E) return;
    const uint s = tiisg / 8, lane = tiisg % 8;
    const uint nb = rank / 32u;
    device const char *row = w_up + (uint64_t)(s * E + d) * nb * 34u;
    float acc = 0.0f;
    for (uint b = 0; b < nb; b++) {
        device const char *blk = row + b * 34u;
        const float scale = (float)*(device const half *)blk;
        const packed_char4 q = *(device const packed_char4 *)(blk + 2u + 4u * lane);
        acc += scale * dot(float4(char4(q)), act[b * 8u + lane]);
    }
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    float g = qwen4_sigmoid(acc) * xn[(uint64_t)tok * E * hc + s * E + d];
    g += simd_shuffle_xor(g, 8);
    g += simd_shuffle_xor(g, 16);
    if (tiisg == 0) mixed[(uint64_t)tok * E + d] = g / (float)hc;
}
