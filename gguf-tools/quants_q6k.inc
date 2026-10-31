/* Q6_K quantization (fork-owned), textually included into quants.c before
 * ds4q_quantize_chunk.  A port of ggml's quantize_row_q6_K_impl and
 * make_qx_quants (MIT): sixteen 16-value sub-blocks per 256-value block,
 * each with a signed 6-bit level search and an int8 scale under one fp16
 * super-scale.  With an imatrix the level search weights each column by its
 * importance, as ggml does.  Used by the Qwen3.8-27B recipe for tensors that
 * need more than 4 bits (docs/QWEN38_27B.md). */

#define DS4Q_GROUP_MAX_EPS 1e-15f

/* ggml make_qx_quants(n, nmax, x, L, rmse_type = 1, qw) */
static float ds4q_make_qx_quants(int n, int nmax, const float *x, int8_t *L, const float *qw) {
    float max = 0.0f, amax = 0.0f;
    for (int i = 0; i < n; i++) {
        const float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < DS4Q_GROUP_MAX_EPS) {
        for (int i = 0; i < n; i++) L[i] = 0;
        return 0.0f;
    }
    float iscale = -nmax / max;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * x[i]);
        l = l < -nmax ? -nmax : l > nmax - 1 ? nmax - 1 : l;
        L[i] = (int8_t)(l + nmax);
        const float w = qw ? qw[i] : x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; is++) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale * x[i]);
            l = l < -nmax ? -nmax : l > nmax - 1 ? nmax - 1 : l;
            const float w = qw ? qw[i] : x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; i++) {
                int l = ds4q_nearest_int(iscale * x[i]);
                L[i] = (int8_t)(nmax + (l < -nmax ? -nmax : l > nmax - 1 ? nmax - 1 : l));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

/* block_q6_K: ql[128] low nibbles, qh[64] high 2-bit pairs, int8 scales[16],
 * fp16 d (210 bytes), ggml's layout. */
static void ds4q_write_q6_k_block(const float *x, uint8_t *y, const float *quant_weights) {
    int8_t L[QK_K];
    float scales[QK_K / 16];
    float max_scale = 0.0f, max_abs_scale = 0.0f;
    for (int ib = 0; ib < QK_K / 16; ib++) {
        const float scale = ds4q_make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib,
                                                quant_weights ? quant_weights + 16 * ib : NULL);
        scales[ib] = scale;
        if (fabsf(scale) > max_abs_scale) { max_abs_scale = fabsf(scale); max_scale = scale; }
    }
    uint8_t *ql = y, *qh = y + QK_K / 2;
    int8_t *sc = (int8_t *)(y + QK_K / 2 + QK_K / 4);
    uint8_t *dp = y + QK_K / 2 + QK_K / 4 + QK_K / 16;
    if (max_abs_scale < DS4Q_GROUP_MAX_EPS) {
        memset(y, 0, 210);
        return;
    }
    const float iscale = -128.0f / max_scale;
    const uint16_t dh = ds4q_f32_to_f16(1.0f / iscale);
    memcpy(dp, &dh, 2);
    const float d0 = ds4q_f16_to_f32(dh);
    for (int ib = 0; ib < QK_K / 16; ib++) {
        const int v = ds4q_nearest_int(iscale * scales[ib]);
        sc[ib] = (int8_t)(v < 127 ? v : 127);
    }
    for (int j = 0; j < QK_K / 16; j++) {
        const float d = d0 * sc[j];
        if (!d) continue;
        for (int ii = 0; ii < 16; ii++) {
            int l = ds4q_nearest_int(x[16 * j + ii] / d);
            l = l < -32 ? -32 : l > 31 ? 31 : l;
            L[16 * j + ii] = (int8_t)(l + 32);
        }
    }
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; l++) {
            const uint8_t q1 = L[j + l + 0] & 0xF, q2 = L[j + l + 32] & 0xF;
            const uint8_t q3 = L[j + l + 64] & 0xF, q4 = L[j + l + 96] & 0xF;
            ql[l + 0] = q1 | (uint8_t)(q3 << 4);
            ql[l + 32] = q2 | (uint8_t)(q4 << 4);
            qh[l] = (uint8_t)((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                              ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
        }
        ql += 64;
        qh += 32;
    }
}

static size_t ds4q_quantize_q6_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q6_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < ncols / QK_K; b++) {
            ds4q_write_q6_k_block(xrow + (size_t)b * QK_K, out + (size_t)row * row_size + (size_t)b * 210,
                                  quant_weights ? quant_weights + (size_t)b * QK_K : NULL);
        }
    }
    return (size_t)nrows * row_size;
}
