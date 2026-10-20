/* Ternary Bonsai 2 kernels (fork-owned).  Appended to the single ds4 Metal
 * library after the upstream sources, so upstream helpers are visible here;
 * every symbol must carry a bonsai2 prefix to avoid collisions.
 *
 * PQ2_0 block: fp16 scale d, then 32 bytes of 2-bit codes, element j in bits
 * 2*(j%4) of byte j/4 (LSB first); weight = (code - 1) * d.  Blocks are 34
 * bytes, so inside a row they are only 2-byte aligned: all weight loads are
 * half/ushort sized.
 *
 * The PQ2_0 GEMM and FWHT shapes follow the PrismML llama.cpp fork
 * (ggml-metal mul_mm.metal kernel_mul_mm, misc.metal kernel_fwht_tg; MIT
 * licensed). */

#define BONSAI2_QK     128
#define BONSAI2_QBYTES 34
#define BONSAI2_HB     1024

/* Must match bonsai2_mm_args in ds4_metal_bonsai2.m. */
struct bonsai2_mm_args {
    uint in_dim;
    uint out_dim;
    uint n_tok;
    uint row_bytes;
};

/* Must match bonsai2_rot_args in ds4_metal_bonsai2.m. */
struct bonsai2_rot_args {
    uint  width;
    uint  perm_hd;
    uint  perm_nk;
    uint  perm_rep;
    uint  write_xn;
    float eps;
};

/* ---------------------------------------------------------------------------
 * PQ2_0 matvec, NT = 1..8 activation rows.
 *
 * Decode streams ~7 GB of weights per token, so a single token must decode
 * 2-bit codes cheaply enough to keep up with DRAM.  Each simdgroup owns NR0
 * output rows; TPB lanes share one 128-weight block (4*BPT weights per lane)
 * and the simdgroup walks 32/TPB blocks per step.
 *
 * fp16 denormal decode: a 32-bit word u holds 16 codes, weights 0-7 in its
 * low half and 8-15 in its high half, code k at bit 2k of its half.  Masking
 * both halves at once, as_type<half2>(u & (0x00030003 << 2k)), yields two
 * fp16 denormals c * 2^(2k-24), so one AND decodes two weights and the
 * half -> float conversion folds into the fma.  Codes 5-7 sit in the
 * exponent bits and are shifted down by 6 first (one shift per word).  The
 * power of two is folded into the activations when they are staged
 * (bonsai2_den_scale, shared by the NR0 rows), and the code offset (-1) is
 * folded in as d * (dot - sum y).  This streams at about the copy rate.
 *
 * For NT > 1 every weight is loaded once and TC tokens are staged per pass.
 * Staging more tokens at once runs out of registers: from three tokens on,
 * one token per pass with 8 rows per simdgroup was measured fastest.
 * ------------------------------------------------------------------------- */
/* 2^(24 - p) for the bit position p a code sits at inside its half after
 * masking (codes 0-4 in place, 5-7 after the >> 6), indexed by weight & 7. */
constant float bonsai2_den_scale[8] = {
    16777216.0f, 4194304.0f, 1048576.0f, 262144.0f, 65536.0f, 1048576.0f, 262144.0f, 65536.0f,
};

template <short NT, short NR0, short BPT, short TCX>
kernel void kernel_bonsai2_mv_pq2(
        constant bonsai2_mm_args &a [[buffer(0)]],
        device const uchar       *w [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *y [[buffer(3)]],
        uint   tg   [[threadgroup_position_in_grid]],
        ushort nsg  [[simdgroups_per_threadgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr short TPB = 32 / BPT;      /* lanes per block */
    constexpr short BPS = 32 / TPB;      /* blocks per simdgroup step */
    constexpr short WPT = 4 * BPT;       /* weights per lane */
    constexpr short NQ  = BPT / 2;       /* ushort code words per lane */
    constexpr short TC  = NT < TCX ? NT : TCX; /* tokens staged per decode pass */

    const uint nb = a.in_dim / BONSAI2_QK;
    const uint row0 = (tg * nsg + sg) * NR0;
    const short ix = lane / TPB;
    const short il = lane % TPB;

    /* Rows past out_dim re-read the last row and are never stored. */
    device const uchar *wr[NR0];
#pragma unroll
    for (short r = 0; r < NR0; r++) {
        wr[r] = w + (ulong)min(row0 + r, a.out_dim - 1) * a.row_bytes + il * BPT;
    }

    float acc[NR0][NT];
#pragma unroll
    for (short r = 0; r < NR0; r++) {
#pragma unroll
        for (short t = 0; t < NT; t++) acc[r][t] = 0.0f;
    }

    for (uint ib = ix; ib < nb; ib += BPS) {
        /* This lane's codes and scale for every row, loaded once and reused
         * by all token passes. */
        ushort q[NR0][NQ];
        float d[NR0];
#pragma unroll
        for (short r = 0; r < NR0; r++) {
            device const uchar *blk = wr[r] + ib * BONSAI2_QBYTES;
            d[r] = *(device const half *)(blk - il * BPT);
#pragma unroll
            for (short j = 0; j < NQ; j++) q[r][j] = ((device const ushort *)(blk + 2))[j];
        }

#pragma unroll
        for (short t0 = 0; t0 < NT; t0 += TC) {
            float c[TC][WPT];
            float sumy[TC];
#pragma unroll
            for (short t = 0; t < TC; t++) {
                device const float4 *xp = (device const float4 *)
                    (x + (ulong)min(t0 + t, NT - 1) * a.in_dim + ib * BONSAI2_QK + il * WPT);
                sumy[t] = 0.0f;
#pragma unroll
                for (short j = 0; j < BPT; j++) {
                    const float4 v = xp[j];
                    sumy[t] += (v.x + v.y) + (v.z + v.w);
#pragma unroll
                    for (short e = 0; e < 4; e++) c[t][4*j + e] = v[e] * bonsai2_den_scale[(4*j + e) & 7];
                }
            }
#pragma unroll
            for (short r = 0; r < NR0; r++) {
                float s[TC];
#pragma unroll
                for (short t = 0; t < TC; t++) s[t] = 0.0f;
#pragma unroll
                for (short j = 0; j < NQ / 2; j++) {
                    const uint u = (uint)q[r][2*j] | ((uint)q[r][2*j + 1] << 16);
                    const uint u6 = u >> 6;
                    half2 h[8];
                    h[0] = as_type<half2>(u  & 0x00030003u);
                    h[1] = as_type<half2>(u  & 0x000C000Cu);
                    h[2] = as_type<half2>(u  & 0x00300030u);
                    h[3] = as_type<half2>(u  & 0x00C000C0u);
                    h[4] = as_type<half2>(u  & 0x03000300u);
                    h[5] = as_type<half2>(u6 & 0x00300030u);
                    h[6] = as_type<half2>(u6 & 0x00C000C0u);
                    h[7] = as_type<half2>(u6 & 0x03000300u);
#pragma unroll
                    for (short t = 0; t < TC; t++) {
                        if (t0 + t >= NT) break;
#pragma unroll
                        for (short k = 0; k < 8; k++) {
                            s[t] = fma((float)h[k].x, c[t][16*j + k], s[t]);
                            s[t] = fma((float)h[k].y, c[t][16*j + 8 + k], s[t]);
                        }
                    }
                }
#pragma unroll
                for (short t = 0; t < TC; t++) {
                    if (t0 + t >= NT) break;
                    acc[r][t0 + t] = fma(d[r], s[t] - sumy[t], acc[r][t0 + t]);
                }
            }
        }
    }

#pragma unroll
    for (short r = 0; r < NR0; r++) {
#pragma unroll
        for (short t = 0; t < NT; t++) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0 && row0 + r < a.out_dim) y[(ulong)t * a.out_dim + row0 + r] = v;
        }
    }
}

typedef decltype(kernel_bonsai2_mv_pq2<1, 4, 4, 2>) bonsai2_mv_pq2_t;

/* NR0/BPT/TC measured on M4 Pro (5120 -> 17408): 4 rows suit one or two
 * tokens; from three tokens on, 8 rows and one token per pass.  The wrapper
 * sizes its grid from these NR0. */
template [[host_name("kernel_bonsai2_mv_pq2_1")]] kernel bonsai2_mv_pq2_t kernel_bonsai2_mv_pq2<1, 4, 4, 2>;
template [[host_name("kernel_bonsai2_mv_pq2_2")]] kernel bonsai2_mv_pq2_t kernel_bonsai2_mv_pq2<2, 4, 4, 2>;
template [[host_name("kernel_bonsai2_mv_pq2_3")]] kernel bonsai2_mv_pq2_t kernel_bonsai2_mv_pq2<3, 8, 4, 1>;
template [[host_name("kernel_bonsai2_mv_pq2_4")]] kernel bonsai2_mv_pq2_t kernel_bonsai2_mv_pq2<4, 8, 4, 1>;
template [[host_name("kernel_bonsai2_mv_pq2_5")]] kernel bonsai2_mv_pq2_t kernel_bonsai2_mv_pq2<5, 8, 4, 1>;
template [[host_name("kernel_bonsai2_mv_pq2_6")]] kernel bonsai2_mv_pq2_t kernel_bonsai2_mv_pq2<6, 8, 4, 1>;
template [[host_name("kernel_bonsai2_mv_pq2_7")]] kernel bonsai2_mv_pq2_t kernel_bonsai2_mv_pq2<7, 8, 4, 1>;
template [[host_name("kernel_bonsai2_mv_pq2_8")]] kernel bonsai2_mv_pq2_t kernel_bonsai2_mv_pq2<8, 8, 4, 1>;

/* ---------------------------------------------------------------------------
 * PQ2_0 GEMM for prefill: a 64 (weight rows) x 32 (tokens) output tile per
 * 128-thread threadgroup, K stepped 32 at a time.  Each step dequantizes a
 * 64x32 weight tile and a 32x32 activation tile to half in threadgroup
 * memory as 8x8 sub-tiles, then each simdgroup accumulates a 32x16 slice
 * with simdgroup_multiply_accumulate in float.  Ternary weights times an
 * fp16 scale are exact in half; only the activations round.
 * ------------------------------------------------------------------------- */
kernel void kernel_bonsai2_mm_pq2(
        constant bonsai2_mm_args &a [[buffer(0)]],
        device const uchar       *w [[buffer(1)]],
        device const float       *x [[buffer(2)]],
        device float             *y [[buffer(3)]],
        threadgroup char     *shmem [[threadgroup(0)]],
        uint2  tg  [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg  [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = 64;
    constexpr short NR1 = 32;
    constexpr short NK  = 32;

    threadgroup half *sa = (threadgroup half *)shmem;          /* NR0 x NK */
    threadgroup half *sb = (threadgroup half *)(shmem + 4096); /* NR1 x NK */

    const uint r0 = tg.y * NR0;
    const uint r1 = tg.x * NR1;
    const short nr0 = (short)min(a.out_dim - r0, (uint)NR0);
    const short nr1 = (short)min(a.n_tok - r1, (uint)NR1);

    /* Loader roles: two threads per weight row (16 k each), four per token
     * row (8 k each).  Past-the-edge rows load a valid row; their results
     * land outside the stored region. */
    const short wrow = tid / 2;
    const short wk = 16 * (tid % 2);
    const short xrow = tid / 4;
    const short xk = 8 * (tid % 4);

    device const uchar *wp = w + (ulong)(r0 + min(wrow, (short)(nr0 - 1))) * a.row_bytes;
    device const float *xp = x + (ulong)(r1 + min(xrow, (short)(nr1 - 1))) * a.in_dim + xk;

    /* Threadgroup offsets of this thread's 16 weights and 8 activations.
     * sa holds 8x8 tiles indexed (k/8)*8 + row/8, each stored k-major;
     * sb holds tiles (k/8)*4 + tok/8, each stored token-major. */
    threadgroup half *sa_dst = sa + 64 * (8 * (wk / 8) + wrow / 8) + wrow % 8;
    threadgroup half *sb_dst = sb + 64 * (4 * (xk / 8) + xrow / 8) + 8 * (xrow % 8);

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint k = 0; k < a.in_dim; k += NK) {
        const uint kw = k + wk;
        device const uchar *blk = wp + (kw / BONSAI2_QK) * BONSAI2_QBYTES;
        const half d = *(device const half *)blk;
        device const ushort *qs = (device const ushort *)(blk + 2 + (kw % BONSAI2_QK) / 4);
        const ushort q0 = qs[0];
        const ushort q1 = qs[1];
        const float4 x0 = *(device const float4 *)(xp + k);
        const float4 x1 = *(device const float4 *)(xp + k + 4);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const ushort q = i < 8 ? q0 : q1;
            const half v = (half)((short)((q >> (2 * (i % 8))) & 3) - 1) * d;
            sa_dst[64 * 8 * (i / 8) + 8 * (i % 8)] = v;
        }
        *(threadgroup half4 *)(sb_dst)     = (half4)x0;
        *(threadgroup half4 *)(sb_dst + 4) = (half4)x1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *la = sa + 4 * 64 * (sg % 2);
        threadgroup const half *lb = sb + 2 * 64 * (sg / 2);
        for (short ik = 0; ik < NK / 8; ik++) {
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], la + 64 * i, 8, 0, false);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lb + 64 * i, 8, 0, false);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            la += 8 * 64;
            lb += 4 * 64;
        }
    }

    /* mc[i] is the 8x8 (token, row) tile at token 16*(sg/2) + 8*(i/4),
     * row 32*(sg%2) + 8*(i%4) of this threadgroup's output tile. */
    const uint orow = r0 + 32 * (sg % 2);
    const uint otok = r1 + 16 * (sg / 2);
    if (nr0 == NR0 && nr1 == NR1) {
        device float *out = y + (ulong)otok * a.out_dim + orow;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], out + 8 * (i % 4) + 8 * (ulong)a.out_dim * (i / 4), a.out_dim, 0, false);
        }
        return;
    }

    /* Edge tile: stage through threadgroup memory and store the valid part. */
    threadgroup float *st = (threadgroup float *)shmem;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], st + 32 * (sg % 2) + 8 * (i % 4) + NR0 * (16 * (sg / 2) + 8 * (i / 4)), NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short e = tid; e < NR0 * NR1; e += 128) {
        const short t = e / NR0;
        const short o = e % NR0;
        if (t < nr1 && o < nr0) y[(ulong)(r1 + t) * a.out_dim + r0 + o] = st[e];
    }
}

/* ---------------------------------------------------------------------------
 * Blockwise Hadamard.  One 256-thread threadgroup transforms a 1024 block;
 * thread tid holds elements i*256 + tid (i = 0..3) in v[i].  Strides 1..16
 * are lane shuffles, 32..128 go through threadgroup memory, 256 and 512 are
 * register pairs.  Butterflies are (a + b, a - b) in natural Sylvester
 * order; the 1/32 normalization is left to the caller.  Ends with a barrier
 * so callers may reuse `sh` straight away.
 * ------------------------------------------------------------------------- */
static inline void bonsai2_fwht1024(thread float v[4], threadgroup float *sh, ushort tid) {
    for (ushort h = 1; h < 32; h <<= 1) {
        for (short i = 0; i < 4; i++) {
            const float o = simd_shuffle_xor(v[i], h);
            v[i] = (tid & h) ? o - v[i] : v[i] + o;
        }
    }
    for (ushort h = 32; h < 256; h <<= 1) {
        for (short i = 0; i < 4; i++) sh[i * 256 + tid] = v[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 4; i++) {
            const float o = sh[i * 256 + (tid ^ h)];
            v[i] = (tid & h) ? o - v[i] : v[i] + o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float a0 = v[0] + v[1], a1 = v[0] - v[1];
    const float a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2;
    v[1] = a1 + a3;
    v[2] = a0 - a2;
    v[3] = a1 - a3;
}

/* y = H(s * P x) for one (block, row).  The GDN permutation reads output
 * head h = r + rep*k from input head k + nk*r. */
kernel void kernel_bonsai2_rotate(
        constant bonsai2_rot_args &a [[buffer(0)]],
        device const float        *x [[buffer(1)]],
        device float              *y [[buffer(2)]],
        device const float        *s [[buffer(3)]],
        uint2  tg  [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[BONSAI2_HB];
    const ulong row = (ulong)tg.y * a.width;
    float v[4];
    for (short i = 0; i < 4; i++) {
        const uint j = tg.x * BONSAI2_HB + i * 256 + tid;
        uint src = j;
        if (a.perm_rep > 1) {
            const uint h = j / a.perm_hd;
            src = j % a.perm_hd + a.perm_hd * (h / a.perm_rep + a.perm_nk * (h % a.perm_rep));
        }
        v[i] = x[row + src] * s[j];
    }
    bonsai2_fwht1024(v, sh, tid);
    for (short i = 0; i < 4; i++) y[row + tg.x * BONSAI2_HB + i * 256 + tid] = v[i] * (1.0f / 32.0f);
}

/* x = s * H(z) in place for one (block, row). */
kernel void kernel_bonsai2_unrotate(
        constant bonsai2_rot_args &a [[buffer(0)]],
        device float              *x [[buffer(1)]],
        device const float        *s [[buffer(2)]],
        uint2  tg  [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[BONSAI2_HB];
    device float *xb = x + (ulong)tg.y * a.width + tg.x * BONSAI2_HB;
    device const float *sb = s + tg.x * BONSAI2_HB;
    float v[4];
    for (short i = 0; i < 4; i++) v[i] = xb[i * 256 + tid];
    bonsai2_fwht1024(v, sh, tid);
    for (short i = 0; i < 4; i++) xb[i * 256 + tid] = v[i] * sb[i * 256 + tid] * (1.0f / 32.0f);
}

/* RMSNorm + rotate for one (block, row).  Every block's threadgroup sums
 * the whole row's squares itself: a 5120 row is 20 KB, cheaper to re-read
 * than to serialize the row on one threadgroup. */
kernel void kernel_bonsai2_rms_rotate(
        constant bonsai2_rot_args &a [[buffer(0)]],
        device const float        *x [[buffer(1)]],
        device float              *y [[buffer(2)]],
        device float             *xn [[buffer(3)]],
        device const float        *w [[buffer(4)]],
        device const float        *s [[buffer(5)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[BONSAI2_HB];
    threadgroup float part[8];
    const ulong row = (ulong)tg.y * a.width;

    float ss = 0.0f;
    device const float4 *x4 = (device const float4 *)(x + row);
    for (uint j = tid; j < a.width / 4; j += 256) {
        const float4 v = x4[j];
        ss += dot(v, v);
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = 0.0f;
    for (short i = 0; i < 8; i++) ss += part[i];
    const float scale = rsqrt(ss / (float)a.width + a.eps);

    const uint b = tg.x * BONSAI2_HB;
    float v[4];
    for (short i = 0; i < 4; i++) {
        const uint j = b + i * 256 + tid;
        const float n = x[row + j] * scale * w[j];
        if (a.write_xn) xn[row + j] = n;
        v[i] = n * s[j];
    }
    bonsai2_fwht1024(v, sh, tid);
    for (short i = 0; i < 4; i++) y[row + b + i * 256 + tid] = v[i] * (1.0f / 32.0f);
}

/* Two BF16 projections of the same small batch (the GDN alpha and beta
 * gates): one simdgroup per output row of either weight, all NT <= 8 tokens
 * accumulated together so each weight row is read once. */
struct bonsai2_gemv2_args {
    uint in_dim;
    uint out_dim;
    uint n_tok;
};

kernel void kernel_bonsai2_gemv_bf16_pair(
        constant bonsai2_gemv2_args &a [[buffer(0)]],
        device const ushort       *wa [[buffer(1)]],
        device const ushort       *wb [[buffer(2)]],
        device const float         *x [[buffer(3)]],
        device float              *ya [[buffer(4)]],
        device float              *yb [[buffer(5)]],
        uint   tg   [[threadgroup_position_in_grid]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg * 4 + sg;
    if (r >= 2 * a.out_dim) return;
    const bool second = r >= a.out_dim;
    const uint o = second ? r - a.out_dim : r;
    device const ushort4 *w4 = (device const ushort4 *)((second ? wb : wa) + (ulong)o * a.in_dim);
    float acc[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (uint k = lane; k < a.in_dim / 4; k += 32) {
        const uint4 u = (uint4)w4[k] << 16;
        const float4 wv = as_type<float4>(u);
        for (uint t = 0; t < a.n_tok; t++) {
            acc[t] += dot(wv, ((device const float4 *)(x + (ulong)t * a.in_dim))[k]);
        }
    }
    device float *y = second ? yb : ya;
    for (uint t = 0; t < a.n_tok; t++) {
        const float v = simd_sum(acc[t]);
        if (lane == 0) y[(ulong)t * a.out_dim + o] = v;
    }
}

/* y = H(s * silu(gate) * up) for one (block, row). */
kernel void kernel_bonsai2_swiglu_rotate(
        constant bonsai2_rot_args &a [[buffer(0)]],
        device const float     *gate [[buffer(1)]],
        device const float       *up [[buffer(2)]],
        device float              *y [[buffer(3)]],
        device const float        *s [[buffer(4)]],
        uint2  tg  [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[BONSAI2_HB];
    const ulong base = (ulong)tg.y * a.width + tg.x * BONSAI2_HB;
    float v[4];
    for (short i = 0; i < 4; i++) {
        const uint j = i * 256 + tid;
        const float g = gate[base + j];
        v[i] = g / (1.0f + exp(-g)) * up[base + j] * s[tg.x * BONSAI2_HB + j];
    }
    bonsai2_fwht1024(v, sh, tid);
    for (short i = 0; i < 4; i++) y[base + i * 256 + tid] = v[i] * (1.0f / 32.0f);
}

/* Gated DeltaNet output + rotate for one (block, row):
 *   y = H(s * P(rms_head(o) * w * silu(z)))
 * o and z are in the scan's tiled V-head order; P is the tiled->grouped
 * permutation of kernel_bonsai2_rotate.  Heads are 128 wide, so a 1024 block
 * holds 8 whole (permuted) heads: thread tid's element i*256 + tid belongs
 * to block head 2i + tid/128, which four simdgroups (sg/4 fixed) share. */
kernel void kernel_bonsai2_gdn_out_rotate(
        constant bonsai2_rot_args &a [[buffer(0)]],
        device const float        *o [[buffer(1)]],
        device const float        *z [[buffer(2)]],
        device float              *y [[buffer(3)]],
        device const float        *w [[buffer(4)]],
        device const float        *s [[buffer(5)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[BONSAI2_HB];
    threadgroup float part[4][8];
    const ulong row = (ulong)tg.y * a.width;
    float v[4];
    uint src[4];
    for (short i = 0; i < 4; i++) {
        const uint j = tg.x * BONSAI2_HB + i * 256 + tid;
        const uint h = j / a.perm_hd;
        src[i] = j % a.perm_hd + a.perm_hd * (h / a.perm_rep + a.perm_nk * (h % a.perm_rep));
        v[i] = o[row + src[i]];
        const float ss = simd_sum(v[i] * v[i]);
        if (lane == 0) part[i][sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const short g = 4 * (sg / 4);
    for (short i = 0; i < 4; i++) {
        const float ss = part[i][g] + part[i][g + 1] + part[i][g + 2] + part[i][g + 3];
        const float zv = z[row + src[i]];
        v[i] *= rsqrt(ss / (float)a.perm_hd + a.eps) * w[src[i] % a.perm_hd] *
                (zv / (1.0f + exp(-zv))) * s[tg.x * BONSAI2_HB + i * 256 + tid];
    }
    bonsai2_fwht1024(v, sh, tid);
    for (short i = 0; i < 4; i++) y[row + tg.x * BONSAI2_HB + i * 256 + tid] = v[i] * (1.0f / 32.0f);
}

/* Gated attention prep without the Qwen3.8 indexer: one simdgroup per
 * (q or kv head, token).  Reuses qwen4_attn_prep_slot (metal/qwen4.metal)
 * with n_idx_head = 0 and a grid of only the H + Hkv head slots; the slot
 * function touches the indexer pointers only for slots >= H + Hkv, so those
 * are passed as aliases of bound buffers and never dereferenced. */
kernel void kernel_bonsai2_attn_prep(
        constant ds4_metal_args_qwen4_attn_prep &args [[buffer(0)]],
        device const float *qg       [[buffer(1)]],  /* [T][H][q(D), gate(D)] */
        device const float *kproj    [[buffer(2)]],  /* [T][Hkv*D] */
        device const float *vproj    [[buffer(3)]],  /* [T][Hkv*D] */
        device const float *g_q      [[buffer(4)]],  /* [D] */
        device const float *g_k      [[buffer(5)]],  /* [D] */
        device float       *q_out    [[buffer(6)]],  /* [T][H*D] */
        device float       *gate_out [[buffer(7)]],  /* [T][H*D] */
        device half        *k_cache  [[buffer(8)]],  /* [cap][Hkv*D] */
        device half        *v_cache  [[buffer(9)]],  /* [cap][Hkv*D] */
        device const uint4 *pos3     [[buffer(10)]], /* [cap] rope positions */
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float row[256];
    const uint pos = args.pos0 + tg.y;
    qwen4_attn_prep_slot(args, tg.x, tg.y, pos, pos3[pos], qg, kproj, vproj, qg, qg, g_q, g_k, g_q,
                         q_out, gate_out, k_cache, v_cache, q_out, q_out, row, lane);
}
