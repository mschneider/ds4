#ifndef DS4_BONSAI2_GPU_H
#define DS4_BONSAI2_GPU_H

/*
 * GPU entry points for the Ternary Bonsai 2 family (fork-owned).
 *
 * Bonsai stores every linear weight as PQ2_0 (ternary, 128 weights per
 * 34-byte block) folded with a blockwise Hadamard rotation:
 *
 *     y = W_stored * (H_blk * (s ⊙ P x))
 *
 * H_blk is the normalized 1024-point Sylvester-Walsh-Hadamard transform
 * applied to every consecutive 1024 block of the activation, s a +-1 sign
 * vector chosen by the input width and P an optional head permutation (GDN
 * ssm_out only).  The rotate kernels below produce the bracketed activation;
 * the PQ2_0 kernels are then plain products with the stored weight.
 *
 * All activations are FP32 row-major [n_tok][width].  Weights live in the
 * mapped GGUF (model_map, model_size, offset), like the upstream wrappers.
 * Sign vectors are FP32 GPU tensors of exactly `width` values.
 */

#include <stdint.h>
#include "ds4_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_BONSAI2_PQ2_BLOCK       128u
#define DS4_BONSAI2_PQ2_BLOCK_BYTES 34u
#define DS4_BONSAI2_HADAMARD_BLOCK  1024u

/* out[t][o] = sum_i W[o][i] * x[t][i] for a PQ2_0 weight with in_dim
 * columns (multiple of 128) and out_dim rows.  Picks a matvec for small
 * n_tok and a simdgroup GEMM for prefill. */
int ds4_gpu_bonsai2_matmul_pq2_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_tok);

/* Forward rotation y = H_blk(s ⊙ P x) over rows of `width` (multiple of
 * 1024).  perm_rep > 1 first applies the GDN tiled->grouped head order:
 * x is viewed as [rep][nk][hd] (hd fastest) and read as [nk][rep][hd], i.e.
 * y_in[hd_i + hd*(r + rep*k)] = x[hd_i + hd*(k + nk*r)].  perm_rep <= 1
 * means no permutation.  x and y must not alias. */
int ds4_gpu_bonsai2_rotate_tensor(
        ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *signs,
        uint32_t n_tok, uint32_t width,
        uint32_t perm_hd, uint32_t perm_nk, uint32_t perm_rep);

/* Inverse rotation for the latent token embedding: x = s ⊙ H_blk(z), in
 * place over n_tok rows of `width`. */
int ds4_gpu_bonsai2_unrotate_tensor(
        ds4_gpu_tensor *x, const ds4_gpu_tensor *signs, uint32_t n_tok, uint32_t width);

/* RMSNorm with a model F32 weight, then the forward rotation:
 *   xn = rms_norm(x) * w            (written when xn != NULL; the BF16
 *                                    GDN alpha/beta projections read it)
 *   y  = H_blk(s ⊙ xn)
 * x is not modified.  No permutation. */
int ds4_gpu_bonsai2_rms_rotate_tensor(
        ds4_gpu_tensor *y, ds4_gpu_tensor *xn, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t norm_offset,
        const ds4_gpu_tensor *signs, uint32_t n_tok, uint32_t width, float eps);

/* ya = Wa x and yb = Wb x for two BF16 weights of the same [out_dim][in_dim]
 * shape (the GDN alpha/beta gates) in one dispatch.  Decode batches only:
 * returns 0 without dispatching when n_tok > 8, so callers fall back. */
int ds4_gpu_bonsai2_gemv_bf16_pair_tensor(
        ds4_gpu_tensor *ya, ds4_gpu_tensor *yb, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t wa_offset, uint64_t wb_offset,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok);

/* SwiGLU then the forward rotation: y = H_blk(s ⊙ (silu(gate) * up)). */
int ds4_gpu_bonsai2_swiglu_rotate_tensor(
        ds4_gpu_tensor *y, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
        const ds4_gpu_tensor *signs, uint32_t n_tok, uint32_t width);

/* Gated DeltaNet output norm and gate, then the ssm_out rotation:
 *   y = H_blk(s ⊙ P(rms_head(o) * norm * silu(z)))
 * o and z are [n_tok][n_rep][n_k_head][head_dim] (the scan's tiled V order);
 * P is the tiled->grouped permutation of the rotate above.  head_dim must be
 * 128 (a Hadamard block then holds whole heads).  o and y must not alias. */
int ds4_gpu_bonsai2_gdn_out_rotate_tensor(
        ds4_gpu_tensor *y, const ds4_gpu_tensor *o, const ds4_gpu_tensor *z,
        const void *model_map, uint64_t model_size, uint64_t norm_offset,
        const ds4_gpu_tensor *signs, uint32_t n_tok, uint32_t head_dim,
        uint32_t n_k_head, uint32_t n_rep, float eps);

/* Gated attention prep without the Qwen3.8 indexer: per head RMSNorm of q
 * (weight q_norm) and k (k_norm), NeoX rope on the first n_rot dims at the
 * positions in pos3 ([cache_cap] uint4 (t, h, w, 0); text uses t = h = w),
 * q -> q_out, the gate half -> gate_out, k and v -> the f16 caches at rows
 * pos0... qg holds per token and head [q(D), gate(D)].  rope_freq holds the
 * n_rot/2 inverse frequencies (ds4.c's g_qwen4_rope_freq) and rope_mscale its
 * attention scale, so YaRN settings stay owned by the shared Qwen3.8 code.
 * Same math and layouts as kernel_qwen4_attn_prep with no indexer heads. */
int ds4_gpu_bonsai2_attn_prep_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out,
        ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const ds4_gpu_tensor *pos3,
        const void *model_map, uint64_t model_size,
        uint64_t q_norm_offset, uint64_t k_norm_offset,
        uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t cache_cap,
        const float *rope_freq, float rope_mscale, float eps);

#ifdef __cplusplus
}
#endif

#endif
