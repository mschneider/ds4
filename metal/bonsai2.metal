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

/* --- plain Qwen3.8-27B ------------------------------------------------- */

/* Must match bonsai2_gdn_out_args in ds4_metal_bonsai2.m. */
struct bonsai2_gdn_out_args {
    uint  n_tok;
    uint  n_head;
    float eps;
};

/* Unrotated GDN output norm and gate, in place in the scan's tiled V order
 * (llama.cpp reorders ssm_out's columns to match, so no permutation):
 *   o = rms_head(o) * norm * silu(z)
 * One simdgroup per (head, token); heads are 128 wide, four values a lane.
 * Same as kernel_qwen4_gdn_out with Qwen3.8-27B's silu gate. */
kernel void kernel_bonsai2_gdn_out_silu(
        constant bonsai2_gdn_out_args &a [[buffer(0)]],
        device float              *o [[buffer(1)]],
        device const float        *z [[buffer(2)]],
        device const float        *w [[buffer(3)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const ulong base = ((ulong)tg.y * a.n_head + tg.x) * 128 + lane * 4;
    float4 v = *(device const float4 *)(o + base);
    const float r = rsqrt(simd_sum(dot(v, v)) / 128.0f + a.eps);
    const float4 zv = *(device const float4 *)(z + base);
    v *= r * *(device const float4 *)(w + lane * 4) * (zv / (1.0f + exp(-zv)));
    *(device float4 *)(o + base) = v;
}

/* Must match bonsai2_q4k_mm_args in ds4_metal_bonsai2.m. */
struct bonsai2_q4k_mm_args {
    uint in_dim;
    uint out_dim;
    uint row_bytes;
    uint n_tok;
};

/* Q4_K product for 1..8 activation rows on fp32 simdgroup matrices, with no
 * threadgroup staging.  The classic matvec spends about as long on its
 * per-weight arithmetic as on streaming the weights, so a second row doubles
 * its time; here every lane decodes its two elements of each 8 x 8 weight
 * fragment straight into registers once, and one multiply covers all eight
 * token columns, so 1 to 8 rows cost about the same.
 *
 * Fragment layout (the one MLX's steel GEMM relies on): lane L holds
 * elements (fm, fn) and (fm, fn + 1) with qid = L / 4,
 * fm = (qid & 4) + (L / 2) % 4, fn = (qid & 2) * 2 + (L % 2) * 2.
 * A = weights [8 rows][8 k], B = x^T [8 k][8 tokens] (tokens past n_tok read
 * as 0), C = [8 rows][8 tokens].  The dot product does not care which k a
 * fragment column stands for, so within a Q4_K group (32 values, value j in
 * byte j of the group's 32-byte run) step st, column c is k = 4c + st: a
 * lane's weights for the whole group are bytes 4fn..4fn+7, one aligned
 * 8-byte load, and its activations x[4fm..4fm+3], one float4 per token.
 * Each simdgroup owns RT tiles of 8 rows and reuses each B fragment for all
 * of them; one tile is fastest on M4 Pro (more cost occupancy).
 *
 * On M4 Pro this costs 0.40-0.49 ms for 5120 -> 17408 at 1-8 rows against
 * 0.21 / 0.40 / 0.60 / 0.79 ms for the classic matvec at 1-4 rows (one pass
 * per row): ds4 uses it from three rows up. */
template <short RT>
kernel void kernel_bonsai2_mm_q4k_rows(
        constant bonsai2_q4k_mm_args &a [[buffer(0)]],
        device const uchar        *w [[buffer(1)]],
        device const float        *x [[buffer(2)]],
        device float              *y [[buffer(3)]],
        uint   tg   [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    const short qid = lane / 4;
    const short fm = (qid & 4) + (lane / 2) % 4;
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const uint row0 = (tg * 4u + sg) * (8u * RT);
    if (row0 >= a.out_dim) return;
    device const uchar *wr[RT];
    for (short r = 0; r < RT; r++) wr[r] = w + (ulong)min(row0 + 8u * r + fm, a.out_dim - 1u) * a.row_bytes;
    const bool v0 = (uint)fn < a.n_tok, v1 = (uint)fn + 1u < a.n_tok;
    device const float4 *x0 = (device const float4 *)(x + (ulong)min((uint)fn, a.n_tok - 1u) * a.in_dim + 4 * fm);
    device const float4 *x1 = (device const float4 *)(x + (ulong)min((uint)fn + 1u, a.n_tok - 1u) * a.in_dim + 4 * fm);
    simdgroup_float8x8 C[RT];
    for (short r = 0; r < RT; r++) C[r] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint nb = a.in_dim / 256u;
    for (uint b = 0; b < nb; b++) {
        /* the block's eight (scale, min) pairs, ggml's get_scale_min_k4 */
        float ds[RT][8], dm[RT][8];
        for (short r = 0; r < RT; r++) {
            device const uchar *blk = wr[r] + b * 144u;
            const float2 dd = float2(*(device const half2 *)blk);
            const uint3 q = *(device const packed_uint3 *)(blk + 4);
            for (short g = 0; g < 8; g++) {
                const uint lo = (g < 4 ? q.x : q.z) >> (8 * (g & 3));
                const uint hi = (g < 4 ? q.y : q.x) >> (8 * (g & 3));
                uint sv, mv;
                if (g < 4) { sv = lo & 63u; mv = (q.y >> (8 * g)) & 63u; }
                else { sv = (lo & 0xFu) | ((hi & 0xC0u) >> 2);
                       mv = ((lo >> 4) & 0xFu) | (((q.y >> (8 * (g - 4))) & 0xC0u) >> 2); }
                ds[r][g] = dd.x * (float)sv;
                dm[r][g] = dd.y * (float)mv;
            }
        }
        for (short g = 0; g < 8; g++) {
            const uint k0 = b * 64u + g * 8u;   /* float4 units */
            const float4 xa = v0 ? x0[k0] : 0.0f;
            const float4 xb = v1 ? x1[k0] : 0.0f;
            uint2 qw[RT];
            for (short r = 0; r < RT; r++) {
                qw[r] = *(device const uint2 *)(wr[r] + b * 144u + 16u + (g >> 1) * 32u + 4u * fn) >> ((g & 1) * 4u);
            }
            for (short st = 0; st < 4; st++) {
                simdgroup_float8x8 B;
                B.thread_elements()[0] = xa[st];
                B.thread_elements()[1] = xb[st];
                for (short r = 0; r < RT; r++) {
                    simdgroup_float8x8 A;
                    A.thread_elements()[0] = ds[r][g] * (float)((qw[r].x >> (8 * st)) & 15u) - dm[r][g];
                    A.thread_elements()[1] = ds[r][g] * (float)((qw[r].y >> (8 * st)) & 15u) - dm[r][g];
                    simdgroup_multiply_accumulate(C[r], A, B, C[r]);
                }
            }
        }
    }
    for (short r = 0; r < RT; r++) {
        const uint row = row0 + 8u * r + fm;
        if (row >= a.out_dim) continue;
        if (v0) y[(ulong)fn * a.out_dim + row] = C[r].thread_elements()[0];
        if (v1) y[(ulong)(fn + 1) * a.out_dim + row] = C[r].thread_elements()[1];
    }
}

#define BONSAI2_Q4K_MM(rt) \
    template [[host_name("kernel_bonsai2_mm_q4k_rows_r" #rt)]] \
    kernel decltype(kernel_bonsai2_mm_q4k_rows<rt>) kernel_bonsai2_mm_q4k_rows<rt>;
BONSAI2_Q4K_MM(1)

/* --- Q6_K (plain Qwen3.8-27B tensors that need more than 4 bits) -------- */

/* Q6_K matvec, one activation row per grid row: llama.cpp's
 * kernel_mul_mv_q6_K_f32_impl (MIT) with two simdgroups of two weight rows
 * and no function constants. */
kernel void kernel_bonsai2_mv_q6k(
        constant bonsai2_q4k_mm_args &a [[buffer(0)]],
        device const char         *w [[buffer(1)]],
        device const float        *x [[buffer(2)]],
        device float              *y [[buffer(3)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR0 = 2;
    constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    const int nb = a.in_dim / 256;
    const uint first_row = (tg.x * 2 + sg) * NR0;
    if (first_row >= a.out_dim) return;
    device const block_q6_K *blk = (device const block_q6_K *)(w + (ulong)first_row * a.row_bytes);
    device const float *yy = x + (ulong)tg.y * a.in_dim;
    float sumf[NR0] = { 0.f };
    float yl[16];
    const short tid = lane / 2, ix = lane % 2, ip = tid / 8, il = tid % 8;
    const short l0 = 4 * il, is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0, q_offset_l = 64 * ip + l0, q_offset_h = 32 * ip + l0;
    for (int i = ix; i < nb; i += 2) {
        device const uint8_t *q1 = blk[i].ql + q_offset_l;
        device const uint8_t *q2 = q1 + 32;
        device const uint8_t *qh = blk[i].qh + q_offset_h;
        device const int8_t *sc = (device const int8_t *)blk[i].scales + is;
        device const half *dh = &blk[i].d;
        device const float *yv = yy + i * 256 + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4 * l + 0] = yv[l + 0];
            yl[4 * l + 1] = yv[l + 32];
            yl[4 * l + 2] = yv[l + 64];
            yl[4 * l + 3] = yv[l + 96];
        }
        for (short row = 0; row < NR0; ++row) {
            float4 sums = 0.f;
            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4 * l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4 * l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4 * l + 2] * ((int8_t)((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4 * l + 3] * ((int8_t)((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
            q1 += a.row_bytes;
            q2 += a.row_bytes;
            qh += a.row_bytes;
            sc += a.row_bytes;
            dh += a.row_bytes / 2;
        }
    }
    for (short row = 0; row < NR0 && first_row + row < a.out_dim; ++row) {
        const float s = simd_sum(sumf[row]);
        if (lane == 0) y[(ulong)tg.y * a.out_dim + first_row + row] = s;
    }
}

/* Prefill: the shared tiled kernel_mul_mm on Q6_K blocks (as the MoE path's
 * kernel_mul_mm_id_q6_K_f32 instantiates it). */
template [[host_name("kernel_bonsai2_mul_mm_q6_K_f32")]] kernel mul_mm_t
kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q6_K, 16, dequantize_q6_K,
              float, float4x4, float, float2x4>;
