/*
 * Metal wrappers for the Ternary Bonsai 2 family (fork-owned).
 *
 * Built on the narrow ds4_metal_internal.h surface only, so upstream changes
 * to ds4_metal.m internals do not touch this file.  Every wrapper joins the
 * active command batch (ds4_gpu_begin_commands/end_commands) when one is open.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>

#include "ds4_metal_internal.h"
#include "ds4_bonsai2_gpu.h"

/* Strong definition overriding the weak default in ds4_metal.m. */
NSArray<NSArray<NSString *> *> *ds4_metal_fork_sources(void) {
    return @[
        @[@"DS4_METAL_BONSAI2_SOURCE", @"metal/bonsai2.metal"],
    ];
}

/* Must match the structs in metal/bonsai2.metal. */
typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_tok;
    uint32_t row_bytes;
} bonsai2_mm_args;

typedef struct {
    uint32_t width;
    uint32_t perm_hd;
    uint32_t perm_nk;
    uint32_t perm_rep;
    uint32_t write_xn;
    float    eps;
} bonsai2_rot_args;

/* Above this many activation rows the matmul switches to the tiled GEMM.
 * One GEMM pass costs about the same for 1..32 tokens (~1.2 ms for
 * 5120 -> 17408 on M4 Pro), which the matvec still beats at 8 tokens. */
#define BONSAI2_MV_MAX_TOK 8u

typedef struct {
    id<MTLBuffer> buf;
    NSUInteger off;
} bonsai2_bind;

static int bonsai2_bind_tensor(bonsai2_bind *b, const ds4_gpu_tensor *t, uint64_t bytes,
                               const char *what) {
    if (!t || ds4_gpu_tensor_bytes(t) < bytes) {
        fprintf(stderr, "ds4: bonsai2 %s tensor is missing or undersized\n", what);
        return 0;
    }
    b->buf = ds4_metal_tensor_buffer(t);
    b->off = ds4_metal_tensor_offset(t);
    return b->buf != nil;
}

static int bonsai2_bind_model(bonsai2_bind *b, const void *map, uint64_t size,
                              uint64_t offset, uint64_t bytes, const char *what) {
    uint64_t inner = 0;
    if (!map || offset > size || bytes > size - offset) {
        fprintf(stderr, "ds4: bonsai2 %s range is outside the mapped model\n", what);
        return 0;
    }
    b->buf = ds4_metal_wrap_model_range(map, size, offset, bytes, &inner);
    b->off = (NSUInteger)inner;
    return b->buf != nil;
}

/* args at buffer(0), binds at buffer(1..n). */
static int bonsai2_dispatch(const char *kernel, const void *args, NSUInteger args_len,
                            const bonsai2_bind *b, int n, MTLSize grid, MTLSize threads,
                            NSUInteger shmem) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = ds4_metal_get_pipeline(kernel);
        if (!pipeline) return 0;
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_metal_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_metal_compute_encoder(cb);
        if (!enc) return 0;
        [enc setComputePipelineState:pipeline];
        [enc setBytes:args length:args_len atIndex:0];
        for (int i = 0; i < n; i++) [enc setBuffer:b[i].buf offset:b[i].off atIndex:(NSUInteger)i + 1];
        if (shmem) [enc setThreadgroupMemoryLength:shmem atIndex:0];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:threads];
        ds4_metal_end_compute_encoder(cb, enc);
        return ds4_metal_finish_command_buffer(cb, owned, kernel);
    }
}

int ds4_gpu_bonsai2_matmul_pq2_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (n_tok == 0 || out_dim == 0) return 1;
    if (in_dim == 0 || in_dim % DS4_BONSAI2_PQ2_BLOCK != 0) {
        fprintf(stderr, "ds4: bonsai2 matmul in_dim %u is not a multiple of %u\n",
                in_dim, DS4_BONSAI2_PQ2_BLOCK);
        return 0;
    }
    if (!ds4_gpu_init()) return 0;
    const uint32_t row_bytes = in_dim / DS4_BONSAI2_PQ2_BLOCK * DS4_BONSAI2_PQ2_BLOCK_BYTES;
    const bonsai2_mm_args args = { in_dim, out_dim, n_tok, row_bytes };
    bonsai2_bind b[3];
    if (!bonsai2_bind_model(&b[0], model_map, model_size, weight_offset,
                            (uint64_t)out_dim * row_bytes, "PQ2_0 weight") ||
        !bonsai2_bind_tensor(&b[1], x, (uint64_t)n_tok * in_dim * sizeof(float), "matmul x") ||
        !bonsai2_bind_tensor(&b[2], out, (uint64_t)n_tok * out_dim * sizeof(float), "matmul out")) {
        return 0;
    }
    if (n_tok > BONSAI2_MV_MAX_TOK) {
        return bonsai2_dispatch("kernel_bonsai2_mm_pq2", &args, sizeof(args), b, 3,
                                MTLSizeMake((n_tok + 31) / 32, (out_dim + 63) / 64, 1),
                                MTLSizeMake(128, 1, 1), 8192);
    }
    static const char *const names[BONSAI2_MV_MAX_TOK] = {
        "kernel_bonsai2_mv_pq2_1", "kernel_bonsai2_mv_pq2_2", "kernel_bonsai2_mv_pq2_3",
        "kernel_bonsai2_mv_pq2_4", "kernel_bonsai2_mv_pq2_5", "kernel_bonsai2_mv_pq2_6",
        "kernel_bonsai2_mv_pq2_7", "kernel_bonsai2_mv_pq2_8",
    };
    /* Four simdgroups per threadgroup; rows per simdgroup match each
     * kernel's NR0 template argument in metal/bonsai2.metal. */
    const uint32_t rows = 4u * (n_tok <= 2 ? 4u : 8u);
    return bonsai2_dispatch(names[n_tok - 1], &args, sizeof(args), b, 3,
                            MTLSizeMake((out_dim + rows - 1) / rows, 1, 1),
                            MTLSizeMake(128, 1, 1), 0);
}

static int bonsai2_check_width(uint32_t width, const char *what) {
    if (width == 0 || width % DS4_BONSAI2_HADAMARD_BLOCK != 0) {
        fprintf(stderr, "ds4: bonsai2 %s width %u is not a multiple of %u\n",
                what, width, DS4_BONSAI2_HADAMARD_BLOCK);
        return 0;
    }
    return ds4_gpu_init();
}

/* One 256-thread threadgroup per (1024 block, row). */
static MTLSize bonsai2_block_grid(uint32_t n_tok, uint32_t width) {
    return MTLSizeMake(width / DS4_BONSAI2_HADAMARD_BLOCK, n_tok, 1);
}

int ds4_gpu_bonsai2_rotate_tensor(
        ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *signs,
        uint32_t n_tok, uint32_t width,
        uint32_t perm_hd, uint32_t perm_nk, uint32_t perm_rep) {
    if (n_tok == 0) return 1;
    if (!bonsai2_check_width(width, "rotate")) return 0;
    if (perm_rep > 1 && (uint64_t)perm_hd * perm_nk * perm_rep != width) {
        fprintf(stderr, "ds4: bonsai2 rotate permutation %ux%ux%u does not cover width %u\n",
                perm_hd, perm_nk, perm_rep, width);
        return 0;
    }
    const uint64_t bytes = (uint64_t)n_tok * width * sizeof(float);
    const bonsai2_rot_args args = { width, perm_hd, perm_nk, perm_rep, 0, 0.0f };
    bonsai2_bind b[3];
    if (!bonsai2_bind_tensor(&b[0], x, bytes, "rotate x") ||
        !bonsai2_bind_tensor(&b[1], y, bytes, "rotate y") ||
        !bonsai2_bind_tensor(&b[2], signs, (uint64_t)width * sizeof(float), "rotate signs")) {
        return 0;
    }
    return bonsai2_dispatch("kernel_bonsai2_rotate", &args, sizeof(args), b, 3,
                            bonsai2_block_grid(n_tok, width), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_bonsai2_unrotate_tensor(
        ds4_gpu_tensor *x, const ds4_gpu_tensor *signs, uint32_t n_tok, uint32_t width) {
    if (n_tok == 0) return 1;
    if (!bonsai2_check_width(width, "unrotate")) return 0;
    const bonsai2_rot_args args = { width, 0, 0, 0, 0, 0.0f };
    bonsai2_bind b[2];
    if (!bonsai2_bind_tensor(&b[0], x, (uint64_t)n_tok * width * sizeof(float), "unrotate x") ||
        !bonsai2_bind_tensor(&b[1], signs, (uint64_t)width * sizeof(float), "unrotate signs")) {
        return 0;
    }
    return bonsai2_dispatch("kernel_bonsai2_unrotate", &args, sizeof(args), b, 2,
                            bonsai2_block_grid(n_tok, width), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_bonsai2_rms_rotate_tensor(
        ds4_gpu_tensor *y, ds4_gpu_tensor *xn, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t norm_offset,
        const ds4_gpu_tensor *signs, uint32_t n_tok, uint32_t width, float eps) {
    if (n_tok == 0) return 1;
    if (!bonsai2_check_width(width, "rms rotate")) return 0;
    const uint64_t bytes = (uint64_t)n_tok * width * sizeof(float);
    const bonsai2_rot_args args = { width, 0, 0, 0, xn != NULL, eps };
    bonsai2_bind b[5];
    if (!bonsai2_bind_tensor(&b[0], x, bytes, "rms rotate x") ||
        !bonsai2_bind_tensor(&b[1], y, bytes, "rms rotate y") ||
        /* Without xn the kernel never touches buffer(3); bind y as a placeholder. */
        !bonsai2_bind_tensor(&b[2], xn ? xn : y, bytes, "rms rotate xn") ||
        !bonsai2_bind_model(&b[3], model_map, model_size, norm_offset,
                            (uint64_t)width * sizeof(float), "rms norm weight") ||
        !bonsai2_bind_tensor(&b[4], signs, (uint64_t)width * sizeof(float), "rms rotate signs")) {
        return 0;
    }
    return bonsai2_dispatch("kernel_bonsai2_rms_rotate", &args, sizeof(args), b, 5,
                            bonsai2_block_grid(n_tok, width), MTLSizeMake(256, 1, 1), 0);
}

/* Must match bonsai2_gemv2_args in metal/bonsai2.metal. */
typedef struct {
    uint32_t in_dim, out_dim, n_tok;
} bonsai2_gemv2_args;

int ds4_gpu_bonsai2_gemv_bf16_pair_tensor(
        ds4_gpu_tensor *ya, ds4_gpu_tensor *yb, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t wa_offset, uint64_t wb_offset,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_tok) {
    if (n_tok == 0 || out_dim == 0) return 1;
    if (n_tok > 8 || in_dim == 0 || in_dim % 4 != 0) return 0;
    if (!ds4_gpu_init()) return 0;
    const bonsai2_gemv2_args args = { in_dim, out_dim, n_tok };
    const uint64_t w_bytes = (uint64_t)out_dim * in_dim * 2u;
    const uint64_t y_bytes = (uint64_t)n_tok * out_dim * sizeof(float);
    bonsai2_bind b[5];
    if (!bonsai2_bind_model(&b[0], model_map, model_size, wa_offset, w_bytes, "bf16 gemv weight a") ||
        !bonsai2_bind_model(&b[1], model_map, model_size, wb_offset, w_bytes, "bf16 gemv weight b") ||
        !bonsai2_bind_tensor(&b[2], x, (uint64_t)n_tok * in_dim * sizeof(float), "bf16 gemv x") ||
        !bonsai2_bind_tensor(&b[3], ya, y_bytes, "bf16 gemv ya") ||
        !bonsai2_bind_tensor(&b[4], yb, y_bytes, "bf16 gemv yb")) {
        return 0;
    }
    return bonsai2_dispatch("kernel_bonsai2_gemv_bf16_pair", &args, sizeof(args), b, 5,
                            MTLSizeMake((2u * out_dim + 3u) / 4u, 1, 1), MTLSizeMake(128, 1, 1), 0);
}

int ds4_gpu_bonsai2_swiglu_rotate_tensor(
        ds4_gpu_tensor *y, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
        const ds4_gpu_tensor *signs, uint32_t n_tok, uint32_t width) {
    if (n_tok == 0) return 1;
    if (!bonsai2_check_width(width, "swiglu rotate")) return 0;
    const uint64_t bytes = (uint64_t)n_tok * width * sizeof(float);
    const bonsai2_rot_args args = { width, 0, 0, 0, 0, 0.0f };
    bonsai2_bind b[4];
    if (!bonsai2_bind_tensor(&b[0], gate, bytes, "swiglu gate") ||
        !bonsai2_bind_tensor(&b[1], up, bytes, "swiglu up") ||
        !bonsai2_bind_tensor(&b[2], y, bytes, "swiglu y") ||
        !bonsai2_bind_tensor(&b[3], signs, (uint64_t)width * sizeof(float), "swiglu signs")) {
        return 0;
    }
    return bonsai2_dispatch("kernel_bonsai2_swiglu_rotate", &args, sizeof(args), b, 4,
                            bonsai2_block_grid(n_tok, width), MTLSizeMake(256, 1, 1), 0);
}

int ds4_gpu_bonsai2_gdn_out_rotate_tensor(
        ds4_gpu_tensor *y, const ds4_gpu_tensor *o, const ds4_gpu_tensor *z,
        const void *model_map, uint64_t model_size, uint64_t norm_offset,
        const ds4_gpu_tensor *signs, uint32_t n_tok, uint32_t head_dim,
        uint32_t n_k_head, uint32_t n_rep, float eps) {
    if (n_tok == 0) return 1;
    const uint32_t width = head_dim * n_k_head * n_rep;
    if (head_dim != 128 || n_rep < 1 || !bonsai2_check_width(width, "gdn out rotate")) {
        fprintf(stderr, "ds4: bonsai2 gdn out rotate needs 128-wide heads\n");
        return 0;
    }
    const uint64_t bytes = (uint64_t)n_tok * width * sizeof(float);
    const bonsai2_rot_args args = { width, head_dim, n_k_head, n_rep, 0, eps };
    bonsai2_bind b[5];
    if (!bonsai2_bind_tensor(&b[0], o, bytes, "gdn out o") ||
        !bonsai2_bind_tensor(&b[1], z, bytes, "gdn out z") ||
        !bonsai2_bind_tensor(&b[2], y, bytes, "gdn out y") ||
        !bonsai2_bind_model(&b[3], model_map, model_size, norm_offset,
                            (uint64_t)head_dim * sizeof(float), "gdn out norm") ||
        !bonsai2_bind_tensor(&b[4], signs, (uint64_t)width * sizeof(float), "gdn out signs")) {
        return 0;
    }
    return bonsai2_dispatch("kernel_bonsai2_gdn_out_rotate", &args, sizeof(args), b, 5,
                            bonsai2_block_grid(n_tok, width), MTLSizeMake(256, 1, 1), 0);
}

/* Must match ds4_metal_args_qwen4_attn_prep in metal/qwen4.metal. */
typedef struct {
    uint32_t n_tokens, n_head, n_head_kv, head_dim, n_rot, n_idx_head, idx_dim, pos0, cache_cap;
    float rope_base, eps;
    uint32_t pad0;
    float rope_mscale;
    float rope_freq[32];
} bonsai2_attn_prep_args;

int ds4_gpu_bonsai2_attn_prep_tensor(
        ds4_gpu_tensor *q_out, ds4_gpu_tensor *gate_out,
        ds4_gpu_tensor *k_cache, ds4_gpu_tensor *v_cache,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kproj, const ds4_gpu_tensor *vproj,
        const ds4_gpu_tensor *pos3,
        const void *model_map, uint64_t model_size,
        uint64_t q_norm_offset, uint64_t k_norm_offset,
        uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t cache_cap,
        const float *rope_freq, float rope_mscale, float eps) {
    if (n_tok == 0) return 1;
    /* The shared slot code keeps D/32 values per lane in 8 registers and
     * ropes at most 64 dims (32 frequencies). */
    if (head_dim < 32 || head_dim > 256 || head_dim % 32 != 0 || n_rot > 64 || n_rot % 2 != 0 ||
        n_rot > head_dim || !rope_freq || n_head_kv == 0 || n_head % n_head_kv != 0 ||
        (uint64_t)pos0 + n_tok > cache_cap) {
        fprintf(stderr, "ds4: bonsai2 attn prep shape is unsupported\n");
        return 0;
    }
    if (!ds4_gpu_init()) return 0;
    bonsai2_attn_prep_args args = {
        n_tok, n_head, n_head_kv, head_dim, n_rot, 0, 0, pos0, cache_cap,
        0.0f, eps, 0, rope_mscale, { 0 },
    };
    for (uint32_t i = 0; i < n_rot / 2; i++) args.rope_freq[i] = rope_freq[i];
    const uint64_t q_bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_tok * n_head_kv * head_dim * sizeof(float);
    const uint64_t cache_bytes = (uint64_t)cache_cap * n_head_kv * head_dim * 2u;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    bonsai2_bind b[10];
    if (!bonsai2_bind_tensor(&b[0], qg, 2u * q_bytes, "attn q/gate projection") ||
        !bonsai2_bind_tensor(&b[1], kproj, kv_bytes, "attn k projection") ||
        !bonsai2_bind_tensor(&b[2], vproj, kv_bytes, "attn v projection") ||
        !bonsai2_bind_model(&b[3], model_map, model_size, q_norm_offset, norm_bytes, "attn q_norm") ||
        !bonsai2_bind_model(&b[4], model_map, model_size, k_norm_offset, norm_bytes, "attn k_norm") ||
        !bonsai2_bind_tensor(&b[5], q_out, q_bytes, "attn q") ||
        !bonsai2_bind_tensor(&b[6], gate_out, q_bytes, "attn gate") ||
        !bonsai2_bind_tensor(&b[7], k_cache, cache_bytes, "k cache") ||
        !bonsai2_bind_tensor(&b[8], v_cache, cache_bytes, "v cache") ||
        !bonsai2_bind_tensor(&b[9], pos3, (uint64_t)cache_cap * 16u, "rope positions")) {
        return 0;
    }
    return bonsai2_dispatch("kernel_bonsai2_attn_prep", &args, sizeof(args), b, 10,
                            MTLSizeMake(n_head + n_head_kv, n_tok, 1), MTLSizeMake(32, 1, 1), 0);
}
