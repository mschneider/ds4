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
#include <stdlib.h>
#include <time.h>
#include <string.h>

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
    uint32_t half_out;
} bonsai2_rot_args;

/* Above this many activation rows the matmul switches to the tiled GEMM.
 * One GEMM pass costs about the same for 1..32 tokens (~1.2 ms for
 * 5120 -> 17408 on M4 Pro), which the matvec still beats at 8 tokens. */
#define BONSAI2_MV_MAX_TOK 8u

typedef struct {
    id<MTLBuffer> buf;
    NSUInteger off;
} bonsai2_bind;

/* Prefill alternative to the fused PQ2_0 GEMM: unpack the weight to F16 once
 * per call, then run upstream's F16 GEMM.  The unpack costs about one extra
 * pass over the F16 matrix, which pays off from about 400 tokens on.  On M4
 * Pro (High Power Mode) whole-model prefill of a 4069-token prompt measured,
 * fused -> unpack: 94.0 -> 96.1 t/s at 512-token chunks, 98.9 at 1024, 100.3
 * at 2048, 100.9 at 4096; a 316-token prompt lost 1.6% and 186 tokens 7.5%.
 * The scratch holds one matrix and is reused by every call.  Bonsai never
 * opens a concurrent encoder, so each unpack waits for the previous GEMM. */
#define BONSAI2_UNPACK_MAX_BYTES (512ull << 20)
static uint32_t g_bonsai2_unpack_min_tok = 2048u;
static id<MTLBuffer> g_bonsai2_unpack;

static void bonsai2_gemm_env(void);

void ds4_gpu_bonsai2_set_unpack_min_tokens(uint32_t n_tok) {
    bonsai2_gemm_env();
    g_bonsai2_unpack_min_tok = n_tok;
}

/* From this many rows on, the rotations write the prefill GEMM inputs as
 * half (the value the GEMM would round them to) and the GEMM reads half. */
static uint32_t g_bonsai2_half_act_min_tok = BONSAI2_MV_MAX_TOK + 1u;

void ds4_gpu_bonsai2_set_half_act_min_tokens(uint32_t n_tok) {
    bonsai2_gemm_env();
    g_bonsai2_half_act_min_tok = n_tok;
}

/* From this many rows on, the prefill GEMM unpacks to F16 and runs
 * upstream's kernel_mul_mm_f16_f32, which uses Metal 4 tensor ops on M5 and
 * later (off by default: on M1-M4 it is the plain simdgroup GEMM the family
 * already beats).  It reads float activations, so half activations stay off
 * from this size on. */
static uint32_t g_bonsai2_upstream_min_tok = UINT32_MAX;

void ds4_gpu_bonsai2_set_upstream_min_tokens(uint32_t n_tok) {
    bonsai2_gemm_env();
    g_bonsai2_upstream_min_tok = n_tok;
}

static int bonsai2_use_upstream(uint32_t n_tok) {
    return n_tok > BONSAI2_MV_MAX_TOK && n_tok >= g_bonsai2_upstream_min_tok;
}

static int bonsai2_act_half(uint32_t n_tok) {
    bonsai2_gemm_env();
    return n_tok > BONSAI2_MV_MAX_TOK && n_tok >= g_bonsai2_half_act_min_tok && !bonsai2_use_upstream(n_tok);
}

void ds4_gpu_bonsai2_release(void) { g_bonsai2_unpack = nil; }

static id<MTLBuffer> bonsai2_unpack_scratch(uint64_t bytes) {
    if (g_bonsai2_unpack && g_bonsai2_unpack.length >= bytes) return g_bonsai2_unpack;
    g_bonsai2_unpack = [ds4_metal_device() newBufferWithLength:(NSUInteger)bytes
                                                       options:MTLResourceStorageModePrivate];
    if (!g_bonsai2_unpack) fprintf(stderr, "ds4: bonsai2 F16 unpack scratch allocation failed\n");
    return g_bonsai2_unpack;
}

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
        if (threads.width * threads.height * threads.depth > pipeline.maxTotalThreadsPerThreadgroup) {
            fprintf(stderr, "ds4: %s allows %lu threads per threadgroup, %lu requested\n", kernel,
                    (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
                    (unsigned long)(threads.width * threads.height * threads.depth));
            return 0;
        }
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

/* Prefill GEMM family (metal/bonsai2.metal kernel_bonsai2_gemm): tile
 * shapes x {PQ2_0 or F16 weights} x {F32 or F16 activations} x {prefetch}. */
typedef struct {
    char name[32];
    uint16_t bm, bn, wm, wn;
    uint8_t aq, bh, pf;
} bonsai2_gemm_cfg;

static const uint16_t g_bonsai2_gemm_tiles[][4] = {
    {64, 32, 2, 2}, {64, 64, 2, 2}, {64, 64, 2, 4}, {128, 32, 4, 2}, {128, 64, 4, 2},
    {64, 128, 2, 4}, {128, 64, 4, 4}, {128, 128, 4, 4}, {32, 64, 1, 2},
};
#define BONSAI2_GEMM_NTILE (sizeof(g_bonsai2_gemm_tiles) / sizeof(g_bonsai2_gemm_tiles[0]))

static int bonsai2_gemm_cfg_get(uint32_t i, bonsai2_gemm_cfg *c) {
    if (i >= BONSAI2_GEMM_NTILE * 8) return 0;
    const uint16_t *t = g_bonsai2_gemm_tiles[i / 8];
    const uint32_t v = i % 8;
    c->bm = t[0]; c->bn = t[1]; c->wm = t[2]; c->wn = t[3];
    c->aq = v < 4; c->bh = (v / 2) % 2; c->pf = v % 2;
    snprintf(c->name, sizeof(c->name), "%ux%u_%ux%u_%c%c%c", c->bm, c->bn, c->wm, c->wn,
             c->aq ? 'q' : 'f', c->bh ? 'h' : 'f', c->pf ? 'p' : 'n');
    return 1;
}

uint32_t ds4_gpu_bonsai2_gemm_cfg_count(void) { return (uint32_t)(BONSAI2_GEMM_NTILE * 8); }

const char *ds4_gpu_bonsai2_gemm_cfg_name(uint32_t i, int *half_x) {
    static char names[BONSAI2_GEMM_NTILE * 8][32];
    bonsai2_gemm_cfg c;
    if (!bonsai2_gemm_cfg_get(i, &c)) return NULL;
    memcpy(names[i], c.name, sizeof(c.name));
    if (half_x) *half_x = c.bh;
    return names[i];
}


/* Runs configuration c; b = { PQ2_0 weight, x, out }.  F16-weight
 * configurations unpack the weight into the shared scratch first. */
static int bonsai2_gemm_run(const bonsai2_gemm_cfg *c, const bonsai2_mm_args *args, bonsai2_bind b[3]) {
    if (!c->aq) {
        const uint64_t f16_bytes = (uint64_t)args->out_dim * args->in_dim * sizeof(uint16_t);
        id<MTLBuffer> scratch = bonsai2_unpack_scratch(f16_bytes);
        if (!scratch) return 0;
        bonsai2_bind ub[2] = { b[0], { scratch, 0 } };
        const uint64_t n_words = (uint64_t)args->out_dim * (args->in_dim / 8);
        if (!bonsai2_dispatch("kernel_bonsai2_unpack_pq2_f16", args, sizeof(*args), ub, 2,
                              MTLSizeMake((NSUInteger)((n_words + 255) / 256), 1, 1),
                              MTLSizeMake(256, 1, 1), 0)) {
            return 0;
        }
        b[0] = (bonsai2_bind){ scratch, 0 };
    }
    char kname[64];
    snprintf(kname, sizeof(kname), "kernel_bonsai2_gemm_%s", c->name);
    const NSUInteger shmem = (NSUInteger)(c->bm + c->bn) * 32u * 2u;
    return bonsai2_dispatch(kname, args, sizeof(*args), b, 3,
                            MTLSizeMake((args->n_tok + c->bn - 1) / c->bn, (args->out_dim + c->bm - 1) / c->bm, 1),
                            MTLSizeMake(32u * c->wm * c->wn, 1, 1), shmem < 4096 ? 4096 : shmem);
}

int ds4_gpu_bonsai2_matmul_cfg_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t in_dim, uint32_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_tok, uint32_t cfg) {
    bonsai2_gemm_cfg c;
    if (!bonsai2_gemm_cfg_get(cfg, &c) || n_tok == 0 || in_dim % DS4_BONSAI2_PQ2_BLOCK != 0) return 0;
    if (!ds4_gpu_init()) return 0;
    const uint32_t row_bytes = in_dim / DS4_BONSAI2_PQ2_BLOCK * DS4_BONSAI2_PQ2_BLOCK_BYTES;
    const bonsai2_mm_args args = { in_dim, out_dim, n_tok, row_bytes };
    bonsai2_bind b[3];
    if (!bonsai2_bind_model(&b[0], model_map, model_size, weight_offset,
                            (uint64_t)out_dim * row_bytes, "PQ2_0 weight") ||
        !bonsai2_bind_tensor(&b[1], x, (uint64_t)n_tok * in_dim * (c.bh ? 2u : 4u), "matmul x") ||
        !bonsai2_bind_tensor(&b[2], out, (uint64_t)n_tok * out_dim * sizeof(float), "matmul out")) {
        return 0;
    }
    return bonsai2_gemm_run(&c, &args, b);
}

/* Production prefill tile.  Auto (the default) picks per call between a
 * narrow and a wide tile: the one with fewer padded token rows, weighted
 * by the wide tile's per-row cost relative to the narrow one.  On M4 Pro
 * 64x64_2x2 costs about 0.91x of 64x32_2x2 per padded token from 96 tokens
 * on, so the narrow tile wins where padding to 64 wastes more: 9-32, 65-96,
 * 129-160, 193-224 and 257-288 tokens.
 * A forced tile (index into g_bonsai2_gemm_tiles) skips the rule. */
static uint32_t g_bonsai2_gemm_tile = UINT32_MAX;   /* auto; see bonsai2_gemm_tile_for */
static uint32_t g_bonsai2_gemm_auto_narrow = 0, g_bonsai2_gemm_auto_wide = 1;
static double   g_bonsai2_gemm_auto_ratio = 0.91;
static uint32_t g_bonsai2_gemm_pf = 0;

static int bonsai2_tile_by_name(const char *s, uint32_t *tile) {
    unsigned bm, bn, wm, wn;
    if (sscanf(s, "%ux%u_%ux%u", &bm, &bn, &wm, &wn) != 4) return 0;
    for (uint32_t i = 0; i < BONSAI2_GEMM_NTILE; i++) {
        const uint16_t *t = g_bonsai2_gemm_tiles[i];
        if (t[0] == bm && t[1] == bn && t[2] == wm && t[3] == wn) { *tile = i; return 1; }
    }
    return 0;
}

static uint32_t bonsai2_env_count(const char *name, uint32_t off_value, uint32_t *dst) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    *dst = strcmp(v, "off") == 0 ? off_value : (uint32_t)strtoul(v, NULL, 10);
    return 1;
}

/* Tuning overrides for other chips (docs/BONSAI2_27B.md, "Tuning the
 * prefill GEMM"), read once before the first prefill GEMM or setter call:
 *   DS4_BONSAI2_GEMM_TILE      auto | BMxBN_WMxWN (one of g_bonsai2_gemm_tiles)
 *   DS4_BONSAI2_GEMM_AUTO      NARROW,WIDE,RATIO for auto, e.g. 64x32_2x2,64x64_2x2,0.91
 *   DS4_BONSAI2_GEMM_PREFETCH  0 | 1
 *   DS4_BONSAI2_UNPACK_MIN     tokens from which weights are unpacked to F16, or off
 *   DS4_BONSAI2_HALF_ACT_MIN   tokens from which GEMM inputs are half, or off
 *   DS4_BONSAI2_UPSTREAM_MIN   tokens from which upstream's F16 GEMM runs (tensor ops on M5+), or off */
static void bonsai2_gemm_env(void) {
    static int done;
    if (done) return;
    done = 1;
    int any = 0;
    const char *v = getenv("DS4_BONSAI2_GEMM_TILE");
    if (v && v[0]) {
        any = 1;
        if (strcmp(v, "auto") != 0 && !bonsai2_tile_by_name(v, &g_bonsai2_gemm_tile))
            fprintf(stderr, "ds4: DS4_BONSAI2_GEMM_TILE=%s is not a known tile; using auto\n", v);
    }
    if ((v = getenv("DS4_BONSAI2_GEMM_AUTO")) && v[0]) {
        any = 1;
        char narrow[32] = "", wide[32] = "";
        double ratio = 0.0;
        uint32_t tn, tw;
        if (sscanf(v, "%31[^,],%31[^,],%lf", narrow, wide, &ratio) == 3 && ratio > 0.0 &&
            bonsai2_tile_by_name(narrow, &tn) && bonsai2_tile_by_name(wide, &tw)) {
            g_bonsai2_gemm_auto_narrow = tn;
            g_bonsai2_gemm_auto_wide = tw;
            g_bonsai2_gemm_auto_ratio = ratio;
        } else {
            fprintf(stderr, "ds4: DS4_BONSAI2_GEMM_AUTO=%s is not NARROW,WIDE,RATIO; ignored\n", v);
        }
    }
    if ((v = getenv("DS4_BONSAI2_GEMM_PREFETCH")) && v[0]) { any = 1; g_bonsai2_gemm_pf = atoi(v) != 0; }
    any |= bonsai2_env_count("DS4_BONSAI2_UNPACK_MIN", 0u, &g_bonsai2_unpack_min_tok);
    any |= bonsai2_env_count("DS4_BONSAI2_HALF_ACT_MIN", UINT32_MAX, &g_bonsai2_half_act_min_tok);
    any |= bonsai2_env_count("DS4_BONSAI2_UPSTREAM_MIN", UINT32_MAX, &g_bonsai2_upstream_min_tok);
    if (any) {
        const uint16_t *n = g_bonsai2_gemm_tiles[g_bonsai2_gemm_auto_narrow];
        const uint16_t *w = g_bonsai2_gemm_tiles[g_bonsai2_gemm_auto_wide];
        char tile[32] = "auto";
        if (g_bonsai2_gemm_tile != UINT32_MAX) {
            const uint16_t *t = g_bonsai2_gemm_tiles[g_bonsai2_gemm_tile];
            snprintf(tile, sizeof(tile), "%ux%u_%ux%u", t[0], t[1], t[2], t[3]);
        }
        fprintf(stderr, "ds4: bonsai2 prefill GEMM: tile %s (auto %ux%u_%ux%u / %ux%u_%ux%u at %.3f), "
                "prefetch %u, unpack from %u, half activations from %u, upstream from %u (tensor ops %s)\n",
                tile, n[0], n[1], n[2], n[3], w[0], w[1], w[2], w[3], g_bonsai2_gemm_auto_ratio,
                g_bonsai2_gemm_pf, g_bonsai2_unpack_min_tok, g_bonsai2_half_act_min_tok,
                g_bonsai2_upstream_min_tok, ds4_metal_tensor_ops_available() ? "on" : "off");
    }
}

void ds4_gpu_bonsai2_set_gemm_tile(uint32_t tile, uint32_t prefetch) {
    bonsai2_gemm_env();
    g_bonsai2_gemm_tile = tile < BONSAI2_GEMM_NTILE ? tile : UINT32_MAX;
    g_bonsai2_gemm_pf = prefetch != 0;
}

static uint32_t bonsai2_gemm_tile_for(uint32_t n_tok) {
    bonsai2_gemm_env();
    if (g_bonsai2_gemm_tile < BONSAI2_GEMM_NTILE) return g_bonsai2_gemm_tile;
    const uint32_t bn = g_bonsai2_gemm_tiles[g_bonsai2_gemm_auto_narrow][1];
    const uint32_t bw = g_bonsai2_gemm_tiles[g_bonsai2_gemm_auto_wide][1];
    const uint32_t pad_n = (n_tok + bn - 1u) / bn * bn, pad_w = (n_tok + bw - 1u) / bw * bw;
    return (double)pad_n < g_bonsai2_gemm_auto_ratio * (double)pad_w ? g_bonsai2_gemm_auto_narrow
                                                                     : g_bonsai2_gemm_auto_wide;
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
        bonsai2_gemm_env();
        const uint64_t f16_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
        if (bonsai2_use_upstream(n_tok) && f16_bytes <= BONSAI2_UNPACK_MAX_BYTES) {
            id<MTLBuffer> scratch = bonsai2_unpack_scratch(f16_bytes);
            if (!scratch) return 0;
            bonsai2_bind ub[2] = { b[0], { scratch, 0 } };
            const uint64_t n_words = (uint64_t)out_dim * (in_dim / 8);
            return bonsai2_dispatch("kernel_bonsai2_unpack_pq2_f16", &args, sizeof(args), ub, 2,
                                    MTLSizeMake((NSUInteger)((n_words + 255) / 256), 1, 1),
                                    MTLSizeMake(256, 1, 1), 0) &&
                   ds4_metal_matmul_f16_buffer(out, scratch, 0, in_dim, out_dim, x, n_tok);
        }
        const int unpack = g_bonsai2_unpack_min_tok && n_tok >= g_bonsai2_unpack_min_tok &&
                           f16_bytes <= BONSAI2_UNPACK_MAX_BYTES;
        const uint32_t v = (unpack ? 4u : 0u) + (bonsai2_act_half(n_tok) ? 2u : 0u) + g_bonsai2_gemm_pf;
        bonsai2_gemm_cfg c;
        if (!bonsai2_gemm_cfg_get(bonsai2_gemm_tile_for(n_tok) * 8u + v, &c)) return 0;
        return bonsai2_gemm_run(&c, &args, b);
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
    const bonsai2_rot_args args = { width, perm_hd, perm_nk, perm_rep, 0, 0.0f, (uint32_t)bonsai2_act_half(n_tok) };
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
    const bonsai2_rot_args args = { width, 0, 0, 0, 0, 0.0f, 0 };
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
    const bonsai2_rot_args args = { width, 0, 0, 0, xn != NULL, eps, (uint32_t)bonsai2_act_half(n_tok) };
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
    const bonsai2_rot_args args = { width, 0, 0, 0, 0, 0.0f, (uint32_t)bonsai2_act_half(n_tok) };
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
    const bonsai2_rot_args args = { width, head_dim, n_k_head, n_rep, 0, eps, (uint32_t)bonsai2_act_half(n_tok) };
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

/* Must match bonsai2_fa_args in metal/bonsai2.metal. */
typedef struct {
    uint32_t n_tokens, n_head, n_head_kv, pos0, tok_per_tg;
    float scale;
} bonsai2_fa_args;

/* Must match BONSAI2_FA_NRT in metal/bonsai2.metal: rows per threadgroup
 * in tiles of 8, two simdgroups per tile. */
#define BONSAI2_FA_NRT 3u

int ds4_gpu_bonsai2_attn_fa_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *k_cache, const ds4_gpu_tensor *v_cache,
        uint32_t n_tok, uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
        uint32_t pos0, uint32_t cache_cap, float scale) {
    if (n_tok == 0) return 1;
    const uint32_t group = n_head_kv ? n_head / n_head_kv : 0;
    if (head_dim != 256 || n_head_kv == 0 || n_head % n_head_kv != 0 || group == 0 ||
        group > 8u * BONSAI2_FA_NRT || (uint64_t)pos0 + n_tok > cache_cap) {
        fprintf(stderr, "ds4: bonsai2 flash attention shape is unsupported\n");
        return 0;
    }
    if (!ds4_gpu_init()) return 0;
    const uint32_t tok_per_tg = 8u * BONSAI2_FA_NRT / group;
    const bonsai2_fa_args args = { n_tok, n_head, n_head_kv, pos0, tok_per_tg, scale };
    const uint64_t q_bytes = (uint64_t)n_tok * n_head * head_dim * sizeof(float);
    const uint64_t cache_bytes = (uint64_t)cache_cap * n_head_kv * head_dim * 2u;
    bonsai2_bind b[5];
    if (!bonsai2_bind_tensor(&b[0], q, q_bytes, "attn q") ||
        !bonsai2_bind_tensor(&b[1], gate, q_bytes, "attn gate") ||
        !bonsai2_bind_tensor(&b[2], k_cache, cache_bytes, "k cache") ||
        !bonsai2_bind_tensor(&b[3], v_cache, cache_bytes, "v cache") ||
        !bonsai2_bind_tensor(&b[4], out, q_bytes, "attn out")) {
        return 0;
    }
    return bonsai2_dispatch("kernel_bonsai2_attn_fa", &args, sizeof(args), b, 5,
                            MTLSizeMake(n_head_kv, (n_tok + tok_per_tg - 1) / tok_per_tg, 1),
                            MTLSizeMake(64u * BONSAI2_FA_NRT, 1, 1), 0);
}

double ds4_gpu_bonsai2_mma_peak(uint32_t iters, uint32_t n_tg, uint32_t nc) {
    if (!ds4_gpu_init()) return 0;
    id<MTLBuffer> buf = [ds4_metal_device() newBufferWithLength:(NSUInteger)n_tg * 4 * 256 * 16 options:MTLResourceStorageModePrivate];
    bonsai2_bind b[1] = { { buf, 0 } };
    const bonsai2_mm_args args = { iters, 0, 0, 0 };
    double best = 1e30;
    for (int r = 0; r < 4; r++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (!ds4_gpu_begin_commands() ||
            !bonsai2_dispatch(nc == 16 ? "kernel_bonsai2_mma_peak16" : "kernel_bonsai2_mma_peak", &args, sizeof(args), b, 1,
                              MTLSizeMake(n_tg, 1, 1), MTLSizeMake(128, 1, 1), 0) ||
            !ds4_gpu_end_commands()) return 0;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        if (r && dt < best) best = dt;
    }
    return (double)n_tg * 4 * iters * nc * 1024 / best * 1e-12;
}
