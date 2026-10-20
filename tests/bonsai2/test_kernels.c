/* Model-free checks of the Ternary Bonsai 2 Metal kernels against double
 * precision CPU references on synthetic data (fixed seeds).  Weights and
 * norm vectors live in an anonymous mapping registered as the "model map",
 * exactly how the engine binds GGUF ranges.
 *
 *   make test-bonsai2-kernels                 correctness
 *   make bench-bonsai2-kernels                decode/prefill throughput
 *   tests/bonsai2/test_kernels --bench-decode  matvec only */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "ds4_gpu.h"
#include "ds4_bonsai2_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

#define QK DS4_BONSAI2_PQ2_BLOCK
#define QB DS4_BONSAI2_PQ2_BLOCK_BYTES
#define HB DS4_BONSAI2_HADAMARD_BLOCK

static int g_failed;

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void report(const char *what, double err, double tol) {
    const int ok = err <= tol;
    printf("  %-44s max err %.3e (tol %.0e) %s\n", what, err, tol, ok ? "ok" : "FAIL");
    if (!ok) g_failed = 1;
}

/* ---- deterministic data ------------------------------------------------ */

static uint64_t g_rng;

static uint64_t rng_next(void) {
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static float rng_uniform(float lo, float hi) {
    return lo + (hi - lo) * (float)((rng_next() >> 40) * (1.0 / 16777216.0));
}

static float *rand_floats(uint64_t n, float lo, float hi) {
    float *v = malloc(n * sizeof(float));
    require_ok(v != NULL, "host allocation");
    for (uint64_t i = 0; i < n; i++) v[i] = rng_uniform(lo, hi);
    return v;
}

static float *rand_signs(uint32_t n) {
    float *v = malloc((uint64_t)n * sizeof(float));
    require_ok(v != NULL, "host allocation");
    for (uint32_t i = 0; i < n; i++) v[i] = (rng_next() & 1) ? 1.0f : -1.0f;
    return v;
}

/* Mostly ternary codes like the real file, plus some code 3 (+2d) so the
 * kernels are checked on the full 2-bit range. */
static void fill_pq2(uint8_t *w, uint64_t n_blocks) {
    for (uint64_t b = 0; b < n_blocks; b++) {
        uint8_t *blk = w + b * QB;
        const _Float16 d = (_Float16)rng_uniform(0.002f, 0.05f);
        memcpy(blk, &d, 2);
        for (int i = 0; i < 32; i++) {
            uint8_t byte = 0;
            for (int j = 0; j < 4; j++) {
                const uint32_t r = (uint32_t)(rng_next() % 100);
                const uint32_t q = r < 32 ? 0 : r < 64 ? 1 : r < 96 ? 2 : 3;
                byte |= (uint8_t)(q << (2 * j));
            }
            blk[2 + i] = byte;
        }
    }
}

/* ---- fake model map ---------------------------------------------------- */

static uint8_t *g_map;
static uint64_t g_map_size, g_map_used;

static void map_create(uint64_t size) {
    g_map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require_ok(g_map != MAP_FAILED, "model map");
    g_map_size = size;
}

/* GGUF aligns tensor data to 32 bytes. */
static uint64_t map_place(uint64_t bytes) {
    const uint64_t off = (g_map_used + 31) & ~31ull;
    require_ok(off + bytes <= g_map_size, "model map placement");
    g_map_used = off + bytes;
    return off;
}

static uint64_t map_pq2(uint32_t in_dim, uint32_t out_dim) {
    const uint64_t blocks = (uint64_t)out_dim * (in_dim / QK);
    const uint64_t off = map_place(blocks * QB);
    fill_pq2(g_map + off, blocks);
    return off;
}

static uint64_t map_floats(const float *v, uint32_t n) {
    const uint64_t off = map_place((uint64_t)n * sizeof(float));
    memcpy(g_map + off, v, (uint64_t)n * sizeof(float));
    return off;
}

/* ---- GPU tensor helpers ------------------------------------------------ */

static ds4_gpu_tensor *upload(const void *data, uint64_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    require_ok(t != NULL, "tensor allocation");
    if (data) require_ok(ds4_gpu_tensor_write(t, 0, data, bytes), "tensor write");
    return t;
}

static float *download(const ds4_gpu_tensor *t, uint64_t n) {
    float *v = malloc(n * sizeof(float));
    require_ok(v != NULL, "host allocation");
    require_ok(ds4_gpu_tensor_read(t, 0, v, n * sizeof(float)), "tensor read");
    return v;
}

/* ---- CPU references ---------------------------------------------------- */

static void pq2_row(const uint8_t *w, uint32_t in_dim, uint32_t o, double *dst) {
    const uint8_t *row = w + (uint64_t)o * (in_dim / QK) * QB;
    for (uint32_t b = 0; b < in_dim / QK; b++) {
        _Float16 d;
        memcpy(&d, row + b * QB, 2);
        for (uint32_t j = 0; j < QK; j++) {
            const int q = (row[b * QB + 2 + j / 4] >> (2 * (j % 4))) & 3;
            dst[b * QK + j] = (double)(q - 1) * (double)d;
        }
    }
}

/* Error of out[t][o] relative to sum |w x|: scale aware even when the dot
 * product cancels.  Checks every row of small outputs and a stride of large
 * ones, always including the tail rows that land in partial tiles. */
static double check_matmul(const uint8_t *w, uint32_t in_dim, uint32_t out_dim,
                           const float *x, uint32_t n_tok, const float *got) {
    const double work = (double)in_dim * out_dim * n_tok;
    const uint32_t stride = work > 1.5e8 ? (uint32_t)(work / 1.5e8) | 1u : 1u;
    double *wr = malloc((uint64_t)in_dim * sizeof(double));
    require_ok(wr != NULL, "host allocation");
    double max_err = 0.0;
    for (uint32_t o = 0; o < out_dim; o++) {
        if (o % stride != 0 && o + 70 < out_dim) continue;
        pq2_row(w, in_dim, o, wr);
        for (uint32_t t = 0; t < n_tok; t++) {
            const float *xr = x + (uint64_t)t * in_dim;
            double s = 0.0, mag = 0.0;
            for (uint32_t i = 0; i < in_dim; i++) {
                s += wr[i] * xr[i];
                mag += fabs(wr[i] * xr[i]);
            }
            const double e = fabs((double)got[(uint64_t)t * out_dim + o] - s) / (mag + 1e-30);
            if (e > max_err) max_err = e;
        }
    }
    free(wr);
    return max_err;
}

/* In-place normalized natural-order FWHT of every 1024 block. */
static void fwht_ref(double *v, uint32_t width) {
    for (uint32_t b = 0; b < width; b += HB) {
        for (uint32_t h = 1; h < HB; h <<= 1) {
            for (uint32_t i = 0; i < HB; i += 2 * h) {
                for (uint32_t j = i; j < i + h; j++) {
                    const double a = v[b + j], c = v[b + j + h];
                    v[b + j] = a + c;
                    v[b + j + h] = a - c;
                }
            }
        }
    }
    for (uint32_t i = 0; i < width; i++) v[i] /= 32.0;
}

static double max_abs(const double *v, uint64_t n) {
    double m = 0.0;
    for (uint64_t i = 0; i < n; i++) if (fabs(v[i]) > m) m = fabs(v[i]);
    return m;
}

/* max |got - want| / max |want| per row, worst row. */
static double row_err(const float *got, const double *want, uint32_t width) {
    double e = 0.0;
    for (uint32_t i = 0; i < width; i++) {
        const double d = fabs((double)got[i] - want[i]);
        if (d > e) e = d;
    }
    return e / (max_abs(want, width) + 1e-30);
}

/* ---- tests ------------------------------------------------------------- */

static void test_matmul(uint32_t in_dim, uint32_t out_dim, const uint32_t *toks, int n) {
    const uint64_t w_off = map_pq2(in_dim, out_dim);
    for (int c = 0; c < n; c++) {
        const uint32_t n_tok = toks[c];
        float *x = rand_floats((uint64_t)n_tok * in_dim, -1.0f, 1.0f);
        ds4_gpu_tensor *tx = upload(x, (uint64_t)n_tok * in_dim * sizeof(float));
        ds4_gpu_tensor *ty = upload(NULL, (uint64_t)n_tok * out_dim * sizeof(float));
        require_ok(ds4_gpu_bonsai2_matmul_pq2_tensor(ty, g_map, g_map_size, w_off, in_dim, out_dim,
                                                     tx, n_tok), "bonsai2 matmul");
        float *y = download(ty, (uint64_t)n_tok * out_dim);
        char what[96];
        snprintf(what, sizeof(what), "matmul %5u -> %6u  n_tok %3u (%s)",
                 in_dim, out_dim, n_tok, n_tok > 8 ? "gemm" : "mv");
        /* Matvec is fp32 throughout; the GEMM rounds activations to half. */
        report(what, check_matmul(g_map + w_off, in_dim, out_dim, x, n_tok, y),
               n_tok > 8 ? 1e-3 : 1e-5);
        free(x);
        free(y);
        ds4_gpu_tensor_free(tx);
        ds4_gpu_tensor_free(ty);
    }
}

static void rotate_ref(double *dst, const float *x, const float *s, uint32_t width,
                       uint32_t hd, uint32_t nk, uint32_t rep) {
    for (uint32_t j = 0; j < width; j++) {
        uint32_t src = j;
        if (rep > 1) {
            const uint32_t h = j / hd;
            src = j % hd + hd * (h / rep + nk * (h % rep));
        }
        dst[j] = (double)x[src] * s[j];
    }
    fwht_ref(dst, width);
}

static void test_rotate(uint32_t width, uint32_t n_tok, uint32_t hd, uint32_t nk, uint32_t rep) {
    const uint64_t n = (uint64_t)n_tok * width;
    float *x = rand_floats(n, -2.0f, 2.0f);
    float *s = rand_signs(width);
    ds4_gpu_tensor *tx = upload(x, n * sizeof(float));
    ds4_gpu_tensor *ts = upload(s, (uint64_t)width * sizeof(float));
    ds4_gpu_tensor *ty = upload(NULL, n * sizeof(float));
    require_ok(ds4_gpu_bonsai2_rotate_tensor(ty, tx, ts, n_tok, width, hd, nk, rep), "bonsai2 rotate");
    float *y = download(ty, n);
    double *ref = malloc(width * sizeof(double));
    double err = 0.0;
    for (uint32_t t = 0; t < n_tok; t++) {
        rotate_ref(ref, x + (uint64_t)t * width, s, width, hd, nk, rep);
        const double e = row_err(y + (uint64_t)t * width, ref, width);
        if (e > err) err = e;
    }
    char what[96];
    snprintf(what, sizeof(what), "rotate width %u n_tok %u perm %ux%ux%u", width, n_tok, hd, nk, rep);
    report(what, err, 1e-5);

    /* Unrotate against its own reference, then as the inverse of rotate. */
    if (rep <= 1) {
        require_ok(ds4_gpu_bonsai2_unrotate_tensor(ty, ts, n_tok, width), "bonsai2 unrotate");
        float *back = download(ty, n);
        double rt = 0.0;
        for (uint32_t t = 0; t < n_tok; t++) {
            for (uint32_t i = 0; i < width; i++) ref[i] = x[(uint64_t)t * width + i];
            const double e = row_err(back + (uint64_t)t * width, ref, width);
            if (e > rt) rt = e;
        }
        snprintf(what, sizeof(what), "unrotate(rotate(x)) width %u n_tok %u", width, n_tok);
        report(what, rt, 1e-5);

        require_ok(ds4_gpu_bonsai2_unrotate_tensor(tx, ts, n_tok, width), "bonsai2 unrotate");
        float *u = download(tx, n);
        double ue = 0.0;
        for (uint32_t t = 0; t < n_tok; t++) {
            for (uint32_t i = 0; i < width; i++) ref[i] = x[(uint64_t)t * width + i];
            fwht_ref(ref, width);
            for (uint32_t i = 0; i < width; i++) ref[i] *= s[i];
            const double e = row_err(u + (uint64_t)t * width, ref, width);
            if (e > ue) ue = e;
        }
        snprintf(what, sizeof(what), "unrotate width %u n_tok %u", width, n_tok);
        report(what, ue, 1e-5);
        free(back);
        free(u);
    }
    free(x);
    free(s);
    free(y);
    free(ref);
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(ts);
    ds4_gpu_tensor_free(ty);
}

static void rms_ref(double *xn, const float *x, const float *w, uint32_t width, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < width; i++) ss += (double)x[i] * x[i];
    const double r = 1.0 / sqrt(ss / width + eps);
    for (uint32_t i = 0; i < width; i++) xn[i] = x[i] * r * w[i];
}

/* rms_rotate with and without xn, then the decode chain rms_rotate ->
 * matmul inside one command batch. */
static void test_rms_rotate(uint32_t width, uint32_t n_tok) {
    const float eps = 1e-6f;
    const uint64_t n = (uint64_t)n_tok * width;
    float *x = rand_floats(n, -3.0f, 3.0f);
    float *w = rand_floats(width, 0.5f, 1.5f);
    float *s = rand_signs(width);
    const uint64_t w_off = map_floats(w, width);
    ds4_gpu_tensor *tx = upload(x, n * sizeof(float));
    ds4_gpu_tensor *ts = upload(s, (uint64_t)width * sizeof(float));
    ds4_gpu_tensor *ty = upload(NULL, n * sizeof(float));
    ds4_gpu_tensor *txn = upload(NULL, n * sizeof(float));
    double *xn_ref = malloc(width * sizeof(double));
    double *y_ref = malloc(width * sizeof(double));

    for (int with_xn = 1; with_xn >= 0; with_xn--) {
        require_ok(ds4_gpu_bonsai2_rms_rotate_tensor(ty, with_xn ? txn : NULL, tx, g_map, g_map_size,
                                                     w_off, ts, n_tok, width, eps), "bonsai2 rms rotate");
        float *y = download(ty, n);
        float *xn = download(txn, n);
        double ey = 0.0, exn = 0.0;
        for (uint32_t t = 0; t < n_tok; t++) {
            rms_ref(xn_ref, x + (uint64_t)t * width, w, width, eps);
            for (uint32_t i = 0; i < width; i++) y_ref[i] = xn_ref[i] * s[i];
            fwht_ref(y_ref, width);
            double e = row_err(y + (uint64_t)t * width, y_ref, width);
            if (e > ey) ey = e;
            e = row_err(xn + (uint64_t)t * width, xn_ref, width);
            if (e > exn) exn = e;
        }
        char what[96];
        snprintf(what, sizeof(what), "rms_rotate width %u n_tok %u%s", width, n_tok, with_xn ? " (+xn)" : "");
        report(what, ey, 1e-5);
        if (with_xn) {
            snprintf(what, sizeof(what), "rms_rotate xn width %u n_tok %u", width, n_tok);
            report(what, exn, 1e-5);
        }
        free(y);
        free(xn);
    }

    const uint32_t out_dim = 248;
    const uint64_t m_off = map_pq2(width, out_dim);
    ds4_gpu_tensor *to = upload(NULL, (uint64_t)n_tok * out_dim * sizeof(float));
    require_ok(ds4_gpu_begin_commands(), "begin commands");
    require_ok(ds4_gpu_bonsai2_rms_rotate_tensor(ty, NULL, tx, g_map, g_map_size, w_off, ts,
                                                 n_tok, width, eps), "batched rms rotate");
    require_ok(ds4_gpu_bonsai2_matmul_pq2_tensor(to, g_map, g_map_size, m_off, width, out_dim, ty, n_tok),
               "batched matmul");
    require_ok(ds4_gpu_end_commands(), "end commands");
    float *rot = malloc(n * sizeof(float));
    for (uint32_t t = 0; t < n_tok; t++) {
        rms_ref(xn_ref, x + (uint64_t)t * width, w, width, eps);
        for (uint32_t i = 0; i < width; i++) y_ref[i] = xn_ref[i] * s[i];
        fwht_ref(y_ref, width);
        for (uint32_t i = 0; i < width; i++) rot[(uint64_t)t * width + i] = (float)y_ref[i];
    }
    float *o = download(to, (uint64_t)n_tok * out_dim);
    char what[96];
    snprintf(what, sizeof(what), "batched rms_rotate -> matmul n_tok %u", n_tok);
    report(what, check_matmul(g_map + m_off, width, out_dim, rot, n_tok, o), 1e-5);

    free(o);
    free(rot);
    free(x);
    free(w);
    free(s);
    free(xn_ref);
    free(y_ref);
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(ts);
    ds4_gpu_tensor_free(ty);
    ds4_gpu_tensor_free(txn);
    ds4_gpu_tensor_free(to);
}

static void test_swiglu_rotate(uint32_t width, uint32_t n_tok) {
    const uint64_t n = (uint64_t)n_tok * width;
    float *g = rand_floats(n, -6.0f, 6.0f);
    float *u = rand_floats(n, -2.0f, 2.0f);
    float *s = rand_signs(width);
    ds4_gpu_tensor *tg = upload(g, n * sizeof(float));
    ds4_gpu_tensor *tu = upload(u, n * sizeof(float));
    ds4_gpu_tensor *ts = upload(s, (uint64_t)width * sizeof(float));
    ds4_gpu_tensor *ty = upload(NULL, n * sizeof(float));
    require_ok(ds4_gpu_bonsai2_swiglu_rotate_tensor(ty, tg, tu, ts, n_tok, width), "bonsai2 swiglu rotate");
    float *y = download(ty, n);
    double *ref = malloc(width * sizeof(double));
    double err = 0.0;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < width; i++) {
            const double gv = g[(uint64_t)t * width + i];
            ref[i] = gv / (1.0 + exp(-gv)) * u[(uint64_t)t * width + i] * s[i];
        }
        fwht_ref(ref, width);
        const double e = row_err(y + (uint64_t)t * width, ref, width);
        if (e > err) err = e;
    }
    char what[96];
    snprintf(what, sizeof(what), "swiglu_rotate width %u n_tok %u", width, n_tok);
    report(what, err, 1e-5);
    free(g);
    free(u);
    free(s);
    free(y);
    free(ref);
    ds4_gpu_tensor_free(tg);
    ds4_gpu_tensor_free(tu);
    ds4_gpu_tensor_free(ts);
    ds4_gpu_tensor_free(ty);
}


/* Paired BF16 gemv (GDN alpha/beta) for decode batches; bf16 weights are
 * the upper halves of random floats, so the reference is exact in double. */
static void test_gemv_bf16_pair(uint32_t n_tok) {
    const uint32_t in_dim = 5120, out_dim = 48;
    const uint64_t nw = (uint64_t)out_dim * in_dim;
    uint16_t *wa = malloc(nw * 2), *wb = malloc(nw * 2);
    for (uint64_t i = 0; i < nw; i++) {
        float fa = rng_uniform(-0.1f, 0.1f), fb = rng_uniform(-0.1f, 0.1f);
        uint32_t ua, ub;
        memcpy(&ua, &fa, 4);
        memcpy(&ub, &fb, 4);
        wa[i] = (uint16_t)(ua >> 16);
        wb[i] = (uint16_t)(ub >> 16);
    }
    const uint64_t off_a = map_place(nw * 2), off_b = map_place(nw * 2);
    memcpy(g_map + off_a, wa, nw * 2);
    memcpy(g_map + off_b, wb, nw * 2);
    float *x = rand_floats((uint64_t)n_tok * in_dim, -2.0f, 2.0f);
    ds4_gpu_tensor *tx = upload(x, (uint64_t)n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *ta = upload(NULL, (uint64_t)n_tok * out_dim * sizeof(float));
    ds4_gpu_tensor *tb = upload(NULL, (uint64_t)n_tok * out_dim * sizeof(float));
    require_ok(ds4_gpu_bonsai2_gemv_bf16_pair_tensor(ta, tb, tx, g_map, g_map_size, off_a, off_b,
                                                     in_dim, out_dim, n_tok), "bonsai2 bf16 gemv pair");
    float *ya = download(ta, (uint64_t)n_tok * out_dim), *yb = download(tb, (uint64_t)n_tok * out_dim);
    double err = 0.0;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            double ra = 0.0, rb = 0.0, mag = 0.0;
            for (uint32_t i = 0; i < in_dim; i++) {
                uint32_t ua = (uint32_t)wa[(uint64_t)o * in_dim + i] << 16;
                uint32_t ub = (uint32_t)wb[(uint64_t)o * in_dim + i] << 16;
                float fa, fb;
                memcpy(&fa, &ua, 4);
                memcpy(&fb, &ub, 4);
                const double xv = x[(uint64_t)t * in_dim + i];
                ra += fa * xv;
                rb += fb * xv;
                mag += fabs(fa * xv) + fabs(fb * xv);
            }
            const double e = fmax(fabs(ya[t * out_dim + o] - ra), fabs(yb[t * out_dim + o] - rb)) / mag;
            if (e > err) err = e;
        }
    }
    char what[96];
    snprintf(what, sizeof(what), "gemv_bf16_pair n_tok %u", n_tok);
    report(what, err, 1e-6);
    require_ok(!ds4_gpu_bonsai2_gemv_bf16_pair_tensor(ta, tb, tx, g_map, g_map_size, off_a, off_b,
                                                      in_dim, out_dim, 9), "bf16 gemv pair refuses n_tok 9");
    free(wa);
    free(wb);
    free(x);
    free(ya);
    free(yb);
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(ta);
    ds4_gpu_tensor_free(tb);
}

/* GDN output: per (tiled) head RMSNorm * w * silu(z), then the permuted
 * rotation, against the unfused reference. */
static void test_gdn_out_rotate(uint32_t n_tok) {
    const uint32_t hd = 128, nk = 16, rep = 3, width = hd * nk * rep;
    const uint64_t n = (uint64_t)n_tok * width;
    const float eps = 1e-6f;
    float *o = rand_floats(n, -0.5f, 0.5f);
    float *z = rand_floats(n, -6.0f, 6.0f);
    float *w = rand_floats(hd, 0.5f, 1.5f);
    float *s = rand_signs(width);
    const uint64_t w_off = map_floats(w, hd);
    ds4_gpu_tensor *to = upload(o, n * sizeof(float));
    ds4_gpu_tensor *tz = upload(z, n * sizeof(float));
    ds4_gpu_tensor *ts = upload(s, (uint64_t)width * sizeof(float));
    ds4_gpu_tensor *ty = upload(NULL, n * sizeof(float));
    require_ok(ds4_gpu_bonsai2_gdn_out_rotate_tensor(ty, to, tz, g_map, g_map_size, w_off, ts,
                                                     n_tok, hd, nk, rep, eps), "bonsai2 gdn out rotate");
    float *y = download(ty, n);
    float *gated = malloc(width * sizeof(float));
    double *ref = malloc(width * sizeof(double));
    double err = 0.0;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float *ot = o + (uint64_t)t * width, *zt = z + (uint64_t)t * width;
        for (uint32_t h = 0; h < nk * rep; h++) {
            double ss = 0.0;
            for (uint32_t i = 0; i < hd; i++) ss += (double)ot[h * hd + i] * ot[h * hd + i];
            const double r = 1.0 / sqrt(ss / hd + eps);
            for (uint32_t i = 0; i < hd; i++) {
                const double zv = zt[h * hd + i];
                gated[h * hd + i] = (float)(ot[h * hd + i] * r * w[i] * zv / (1.0 + exp(-zv)));
            }
        }
        rotate_ref(ref, gated, s, width, hd, nk, rep);
        const double e = row_err(y + (uint64_t)t * width, ref, width);
        if (e > err) err = e;
    }
    char what[96];
    snprintf(what, sizeof(what), "gdn_out_rotate n_tok %u", n_tok);
    report(what, err, 1e-5);
    free(o);
    free(z);
    free(w);
    free(s);
    free(y);
    free(gated);
    free(ref);
    ds4_gpu_tensor_free(to);
    ds4_gpu_tensor_free(tz);
    ds4_gpu_tensor_free(ts);
    ds4_gpu_tensor_free(ty);
}
static void head_norm_rope(double *dst, const float *src, const float *w, uint32_t D, uint32_t n_rot,
                           uint32_t pos, const float *freq, float mscale, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < D; i++) ss += (double)src[i] * src[i];
    const double r = 1.0 / sqrt(ss / D + eps);
    for (uint32_t i = 0; i < D; i++) dst[i] = src[i] * r * w[i];
    for (uint32_t i = 0; i < n_rot / 2; i++) {
        const double th = (double)pos * freq[i];
        const double c = cos(th) * mscale, sn = sin(th) * mscale;
        const double a = dst[i], b = dst[i + n_rot / 2];
        dst[i] = a * c - b * sn;
        dst[i + n_rot / 2] = a * sn + b * c;
    }
}

static void test_attn_prep(uint32_t n_tok) {
    const uint32_t H = 24, Hkv = 4, D = 256, n_rot = 64, pos0 = 37, cap = pos0 + n_tok + 5;
    const float eps = 1e-6f, mscale = 1.07f;
    float freq[32];
    for (uint32_t i = 0; i < n_rot / 2; i++) freq[i] = (float)pow(10000000.0, -2.0 * i / n_rot);

    float *qg = rand_floats((uint64_t)n_tok * H * 2 * D, -2.0f, 2.0f);
    float *kp = rand_floats((uint64_t)n_tok * Hkv * D, -2.0f, 2.0f);
    float *vp = rand_floats((uint64_t)n_tok * Hkv * D, -2.0f, 2.0f);
    float *gq = rand_floats(D, 0.5f, 1.5f);
    float *gk = rand_floats(D, 0.5f, 1.5f);
    const uint64_t gq_off = map_floats(gq, D), gk_off = map_floats(gk, D);
    uint32_t *pos3 = calloc((uint64_t)cap * 4, sizeof(uint32_t));
    for (uint32_t p = 0; p < cap; p++) pos3[4 * p] = pos3[4 * p + 1] = pos3[4 * p + 2] = p;
    const uint64_t cache_bytes = (uint64_t)cap * Hkv * D * 2;
    uint16_t *zero = calloc(cache_bytes, 1);

    ds4_gpu_tensor *tqg = upload(qg, (uint64_t)n_tok * H * 2 * D * sizeof(float));
    ds4_gpu_tensor *tkp = upload(kp, (uint64_t)n_tok * Hkv * D * sizeof(float));
    ds4_gpu_tensor *tvp = upload(vp, (uint64_t)n_tok * Hkv * D * sizeof(float));
    ds4_gpu_tensor *tp3 = upload(pos3, (uint64_t)cap * 16);
    ds4_gpu_tensor *tq = upload(NULL, (uint64_t)n_tok * H * D * sizeof(float));
    ds4_gpu_tensor *tgo = upload(NULL, (uint64_t)n_tok * H * D * sizeof(float));
    ds4_gpu_tensor *tkc = upload(zero, cache_bytes);
    ds4_gpu_tensor *tvc = upload(zero, cache_bytes);
    require_ok(ds4_gpu_bonsai2_attn_prep_tensor(tq, tgo, tkc, tvc, tqg, tkp, tvp, tp3, g_map, g_map_size,
                                                gq_off, gk_off, n_tok, H, Hkv, D, n_rot, pos0, cap,
                                                freq, mscale, eps), "bonsai2 attn prep");
    float *q = download(tq, (uint64_t)n_tok * H * D);
    float *gate = download(tgo, (uint64_t)n_tok * H * D);
    _Float16 *kc = malloc(cache_bytes), *vc = malloc(cache_bytes);
    require_ok(ds4_gpu_tensor_read(tkc, 0, kc, cache_bytes) && ds4_gpu_tensor_read(tvc, 0, vc, cache_bytes),
               "cache read");

    double ref[256], eq = 0.0, eg = 0.0, ek = 0.0, ev = 0.0;
    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t pos = pos0 + t;
        for (uint32_t h = 0; h < H; h++) {
            const float *src = qg + ((uint64_t)t * H + h) * 2 * D;
            head_norm_rope(ref, src, gq, D, n_rot, pos, freq, mscale, eps);
            double e = row_err(q + ((uint64_t)t * H + h) * D, ref, D);
            if (e > eq) eq = e;
            for (uint32_t i = 0; i < D; i++) ref[i] = src[D + i];
            e = row_err(gate + ((uint64_t)t * H + h) * D, ref, D);
            if (e > eg) eg = e;
        }
        for (uint32_t h = 0; h < Hkv; h++) {
            head_norm_rope(ref, kp + ((uint64_t)t * Hkv + h) * D, gk, D, n_rot, pos, freq, mscale, eps);
            const _Float16 *kr = kc + ((uint64_t)pos * Hkv + h) * D;
            const _Float16 *vr = vc + ((uint64_t)pos * Hkv + h) * D;
            for (uint32_t i = 0; i < D; i++) {
                double e = fabs((double)kr[i] - ref[i]) / (fabs(ref[i]) + 1e-2);
                if (e > ek) ek = e;
                const double vv = vp[((uint64_t)t * Hkv + h) * D + i];
                e = fabs((double)vr[i] - vv) / (fabs(vv) + 1e-2);
                if (e > ev) ev = e;
            }
        }
    }
    /* Rows outside pos0..pos0+n_tok-1 must stay untouched. */
    int stray = 0;
    for (uint64_t i = 0; i < (uint64_t)cap * Hkv * D; i++) {
        const uint64_t p = i / ((uint64_t)Hkv * D);
        if ((p < pos0 || p >= pos0 + n_tok) && (kc[i] != 0 || vc[i] != 0)) stray = 1;
    }
    char what[96];
    snprintf(what, sizeof(what), "attn_prep q    n_tok %2u pos0 %u", n_tok, pos0);
    report(what, eq, 2e-5);
    snprintf(what, sizeof(what), "attn_prep gate n_tok %2u", n_tok);
    report(what, eg, 0.0);
    snprintf(what, sizeof(what), "attn_prep k f16 cache n_tok %2u", n_tok);
    report(what, ek, 1e-3);
    snprintf(what, sizeof(what), "attn_prep v f16 cache n_tok %2u", n_tok);
    report(what, ev, 1e-3);
    snprintf(what, sizeof(what), "attn_prep other cache rows untouched");
    report(what, stray, 0.0);

    free(qg); free(kp); free(vp); free(gq); free(gk); free(pos3); free(zero);
    free(q); free(gate); free(kc); free(vc);
    ds4_gpu_tensor_free(tqg); ds4_gpu_tensor_free(tkp); ds4_gpu_tensor_free(tvp);
    ds4_gpu_tensor_free(tp3); ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tgo);
    ds4_gpu_tensor_free(tkc); ds4_gpu_tensor_free(tvc);
}

/* ---- benchmark --------------------------------------------------------- */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Cycles through enough weight copies (>= 512 MiB) that the system level
 * cache cannot serve repeats, so GB/s is DRAM streaming bandwidth. */
static void bench_matmul(uint32_t in_dim, uint32_t out_dim, uint32_t n_tok, int iters) {
    const uint64_t wbytes = (uint64_t)out_dim * (in_dim / QK) * QB;
    uint32_t copies = (uint32_t)((512ull << 20) / wbytes) + 1;
    if (copies > 32) copies = 32;
    uint64_t offs[32];
    g_map_used = 0;
    for (uint32_t c = 0; c < copies; c++) offs[c] = map_pq2(in_dim, out_dim);
    float *x = rand_floats((uint64_t)n_tok * in_dim, -1.0f, 1.0f);
    ds4_gpu_tensor *tx = upload(x, (uint64_t)n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *ty = upload(NULL, (uint64_t)n_tok * out_dim * sizeof(float));

    double best = 1e30;
    for (int rep = 0; rep < 4; rep++) {
        require_ok(ds4_gpu_begin_commands(), "begin commands");
        const double t0 = now_s();
        for (int i = 0; i < iters; i++) {
            require_ok(ds4_gpu_bonsai2_matmul_pq2_tensor(ty, g_map, g_map_size, offs[i % copies], in_dim,
                                                         out_dim, tx, n_tok), "bench matmul");
        }
        require_ok(ds4_gpu_end_commands(), "end commands");
        const double dt = (now_s() - t0) / iters;
        if (rep > 0 && dt < best) best = dt;   /* rep 0 warms pipelines and pages */
    }
    printf("  %5u -> %6u  n_tok %3u: %8.3f ms  %7.1f GB/s  %8.1f GFLOP/s\n",
           in_dim, out_dim, n_tok, best * 1e3, wbytes / best * 1e-9,
           2.0 * in_dim * out_dim * n_tok / best * 1e-9);
    free(x);
    ds4_gpu_tensor_free(tx);
    ds4_gpu_tensor_free(ty);
}

int main(int argc, char **argv) {
    const int decode_only = argc > 1 && strcmp(argv[1], "--bench-decode") == 0;
    const int bench = decode_only || (argc > 1 && strcmp(argv[1], "--bench") == 0);
    require_ok(ds4_gpu_init(), "GPU initialization");
    map_create(bench ? (1100ull << 20) : (512ull << 20));
    g_rng = 0xb0d5a12ull;
    if (bench) {
        require_ok(ds4_gpu_set_model_map(g_map, g_map_size), "model map registration");
        printf("bonsai2 kernel bench (decode matvec, prefill gemm):\n");
        bench_matmul(5120, 17408, 1, 64);
        bench_matmul(17408, 5120, 1, 64);
        bench_matmul(6144, 5120, 1, 64);
        bench_matmul(5120, 248320, 1, 8);
        bench_matmul(5120, 17408, 2, 64);
        bench_matmul(5120, 17408, 4, 64);
        bench_matmul(5120, 17408, 8, 64);
        if (!decode_only) {
            bench_matmul(5120, 17408, 512, 8);
            bench_matmul(17408, 5120, 512, 8);
        }
        ds4_gpu_cleanup();
        return 0;
    }
    require_ok(ds4_gpu_set_model_map(g_map, g_map_size), "model map registration");

    printf("bonsai2 kernels:\n");
    const uint32_t all[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 64, 333 };
    const uint32_t some[] = { 1, 4, 8, 64, 333 };
    const uint32_t down[] = { 1, 3, 6, 9, 333 };
    const uint32_t one[] = { 1 };
    test_matmul(5120, 248, all, 12);
    test_matmul(6144, 5120, all, 12);
    test_matmul(5120, 10240, some, 5);
    test_matmul(17408, 5120, down, 5);
    test_matmul(5120, 248320, one, 1);

    test_rotate(6144, 3, 128, 16, 3);
    test_rotate(6144, 3, 0, 0, 0);
    test_rotate(17408, 2, 0, 0, 0);
    test_rms_rotate(5120, 3);
    test_rms_rotate(5120, 1);
    test_swiglu_rotate(17408, 2);
    test_gdn_out_rotate(1);
    test_gdn_out_rotate(5);
    test_gemv_bf16_pair(1);
    test_gemv_bf16_pair(8);
    test_attn_prep(1);
    test_attn_prep(7);
    test_attn_prep(40);

    ds4_gpu_cleanup();
    munmap(g_map, g_map_size);
    if (g_failed) {
        printf("bonsai2 kernels: FAILED\n");
        return 1;
    }
    printf("bonsai2 kernels: all checks passed\n");
    return 0;
}
