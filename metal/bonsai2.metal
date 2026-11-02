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
    uint  half_out;   /* y holds half (clamped to the fp16 range): prefill GEMM input */
};

/* Rotation output: float, or half exactly as the GEMMs round a float input. */
static inline void bonsai2_put(device float *y, ulong i, float v, uint half_out) {
    if (half_out) ((device half *)y)[i] = (half)clamp(v, -65504.0f, 65504.0f);
    else y[i] = v;
}

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
 * PQ2_0 -> F16 unpack for the prefill GEMM.  One thread expands one 16-bit
 * word (8 weights) into rows of in_dim halves, the layout the F16-weight
 * GEMM variants and upstream's kernel_mul_mm_f16_f32 read.  Values are
 * (code - 1) * d computed in half, the same exact values the PQ2_0 variants
 * stage.  Rows are whole
 * blocks, so the global block index walks the matrix contiguously.
 * ------------------------------------------------------------------------- */
kernel void kernel_bonsai2_unpack_pq2_f16(
        constant bonsai2_mm_args &a [[buffer(0)]],
        device const uchar       *w [[buffer(1)]],
        device half4             *y [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint n_words = a.out_dim * (a.in_dim / 8);
    if (gid >= n_words) return;
    const uint blk = gid / 16, k = gid % 16;
    device const uchar *bp = w + (ulong)blk * BONSAI2_QBYTES;
    const half d = *(device const half *)bp;
    const ushort q = ((device const ushort *)(bp + 2))[k];
    half4 lo, hi;
    for (short i = 0; i < 4; i++) {
        lo[i] = (half)((short)((q >> (2 * i)) & 3) - 1) * d;
        hi[i] = (half)((short)((q >> (2 * (i + 4))) & 3) - 1) * d;
    }
    y[2 * (ulong)gid]     = lo;
    y[2 * (ulong)gid + 1] = hi;
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
    for (short i = 0; i < 4; i++) bonsai2_put(y, row + tg.x * BONSAI2_HB + i * 256 + tid, v[i] * (1.0f / 32.0f), a.half_out);
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
    for (short i = 0; i < 4; i++) bonsai2_put(y, row + b + i * 256 + tid, v[i] * (1.0f / 32.0f), a.half_out);
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
    for (short i = 0; i < 4; i++) bonsai2_put(y, base + i * 256 + tid, v[i] * (1.0f / 32.0f), a.half_out);
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
    for (short i = 0; i < 4; i++) bonsai2_put(y, row + tg.x * BONSAI2_HB + i * 256 + tid, v[i] * (1.0f / 32.0f), a.half_out);
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

/* Prefill attention for Bonsai's grouped heads (H/Hkv query heads per KV
 * head), flash style: one threadgroup per (kv head, block of query tokens).
 * Its rows are (token, query head) pairs, token-major, in 3 tiles of 8 (4
 * tokens x 6 heads for Bonsai), so a K/V tile staged in threadgroup memory
 * serves every query token of the block instead of being re-read per token
 * as in kernel_qwen4_attn_mm.  The inner loop is that kernel's: keys in tiles
 * of 16, queries rounded to half, simdgroup 2*rt + dh owns row tile rt and
 * dim half dh (16 accumulators), the two simdgroups of a row tile each
 * score one key half, exchange the scores and run the same online softmax,
 * so the output is bit-identical.  Keys run to the last token of the block;
 * per row, keys after its own position are masked.
 *
 * Shape study on M4 Pro (one layer, 4096 tokens; TFLOPS of causal work):
 * queries in registers instead of threadgroup memory 1.5 (register
 * pressure), 48 rows 1.5-1.6, K and V staged together 1.6, 16 rows 2.4,
 * 32 rows 2.9, this kernel 3.3.  K, then V, share one 8 KB buffer. */
#define BONSAI2_FA_KT  16
#define BONSAI2_FA_NRT 3

struct bonsai2_fa_args {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t pos0;
    uint32_t tok_per_tg;   /* query tokens per threadgroup, tok_per_tg * group <= 8 * NRT */
    float    scale;
};

kernel void kernel_bonsai2_attn_fa(
        constant bonsai2_fa_args &args [[buffer(0)]],
        device const float *q          [[buffer(1)]],  /* [T][H*D] */
        device const float *gate       [[buffer(2)]],  /* [T][H*D] */
        device const half  *k_cache    [[buffer(3)]],  /* [cap][Hkv*D] */
        device const half  *v_cache    [[buffer(4)]],  /* [cap][Hkv*D] */
        device float       *out        [[buffer(5)]],  /* [T][H*D] */
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint D = 256, KT = BONSAI2_FA_KT, NRT = BONSAI2_FA_NRT, NT = 64 * NRT, R = 8 * NRT;
    const uint H = args.n_head, Hkv = args.n_head_kv, group = H / Hkv;
    const uint kvh = tgpig.x, tok0 = tgpig.y * args.tok_per_tg;
    if (kvh >= Hkv || tok0 >= args.n_tokens) return;
    const uint ntok = min(args.tok_per_tg, args.n_tokens - tok0);
    const uint nrows = ntok * group;
    const uint n = args.pos0 + tok0 + ntok;               /* keys 0..n-1 */
    const uint rt = sgitg >> 1, dh = sgitg & 1u;
    const uint lr = tiisg >> 2, lc = (tiisg & 3u) * 4;    /* this lane's row and first of four key columns */
    const uint row = rt * 8 + lr;
    const bool row_ok = row < nrows;
    const uint qpos = args.pos0 + tok0 + (row_ok ? row / group : 0);

    threadgroup half  KV[KT * D];                         /* K tile, then V tile */
    threadgroup half  Qs[R * D];                          /* scaled queries as half */
    /* Keep these aliases: addressing the K and V tiles through them instead
     * of KV directly measured 3.3 vs 2.95 TFLOPS (Metal codegen). */
    threadgroup half *Ks = KV, *Vs = KV;
    threadgroup float Sx[NRT][2][64];                     /* [row tile][key half] scores */
    threadgroup half  Ps[2 * NRT][128];                   /* per simdgroup 8 x 16 probabilities */
    threadgroup float Dg[2 * NRT][64];                    /* per simdgroup diagonal factors / output tile */
    threadgroup float Id[64];

    for (uint i = tid; i < R * D; i += NT) {
        const uint r = i / D, d = i % D;
        half v = 0.0h;
        if (r < nrows) {
            const uint t = tok0 + r / group, h = kvh * group + r % group;
            v = (half)(q[((uint64_t)t * H + h) * D + d] * args.scale);
        }
        Qs[i] = v;
    }
    if (tid < 64) Id[tid] = (tid >> 3) == (tid & 7u) ? 1.0f : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 I;
    simdgroup_load(I, Id, 8, 0, false);
    simdgroup_float8x8 O[16];
#pragma unroll
    for (uint j = 0; j < 16; j++) O[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m_row = -3.0e38f, l_row = 0.0f;

    for (uint t0 = 0; t0 < n; t0 += KT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < KT * D / 8; i += NT) {     /* 16 rows of K as uint4 */
            const uint p = t0 + i / (D / 8);
            const uint64_t r8 = (((uint64_t)p * Hkv + kvh) * D) / 8 + i % (D / 8);
            ((threadgroup uint4 *)Ks)[i] = p < n ? ((device const uint4 *)k_cache)[r8] : uint4(0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* scores of this row tile against key half dh, four independent chains */
        simdgroup_float8x8 S[4];
#pragma unroll
        for (uint i = 0; i < 4; i++) S[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma unroll
        for (uint kk = 0; kk < 8; kk++) {
#pragma unroll
            for (uint i = 0; i < 4; i++) {
                simdgroup_half8x8 Qt, Kt;
                simdgroup_load(Qt, Qs + (rt * 8) * D + (kk * 4 + i) * 8, D, 0, false);
                simdgroup_load(Kt, Ks + (dh * 8) * D + (kk * 4 + i) * 8, D, 0, true);
                simdgroup_multiply_accumulate(S[i], Qt, Kt, S[i]);
            }
        }
#pragma unroll
        for (uint i = 1; i < 4; i++) simdgroup_multiply_accumulate(S[0], I, S[i], S[0]);
        simdgroup_store(S[0], Sx[rt][dh], 8, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < KT * D / 8; i += NT) {     /* every simdgroup is past K: V takes its place */
            const uint p = t0 + i / (D / 8);
            const uint64_t r8 = (((uint64_t)p * Hkv + kvh) * D) / 8 + i % (D / 8);
            ((threadgroup uint4 *)Vs)[i] = p < n ? ((device const uint4 *)v_cache)[r8] : uint4(0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sv[4], pv[4];
        bool valid[4];
        float mx = -3.0e38f;
#pragma unroll
        for (uint c = 0; c < 4; c++) {
            const uint key = lc + c, p = t0 + key;
            valid[c] = row_ok && p < n && p <= qpos;
            sv[c] = valid[c] ? Sx[rt][key >> 3][lr * 8 + (key & 7u)] : -3.0e38f;
            mx = max(mx, sv[c]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float m_new = max(m_row, mx);
        const float corr = exp(m_row - m_new);
        float rs = 0.0f;
#pragma unroll
        for (uint c = 0; c < 4; c++) {
            pv[c] = valid[c] ? exp(sv[c] - m_new) : 0.0f;
            rs += pv[c];
            Ps[sgitg][lr * 16 + lc + c] = (half)pv[c];
        }
        rs += simd_shuffle_xor(rs, 1);
        rs += simd_shuffle_xor(rs, 2);
        l_row = l_row * corr + rs;
        m_row = m_new;
        const bool rescale = simd_any(corr != 1.0f);
        if (rescale) {
            for (uint i = tiisg; i < 64; i += 32) Dg[sgitg][i] = (i >> 3) == (i & 7u) ? simd_shuffle(corr, (ushort)((i >> 3) * 4)) : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (rescale) {
            simdgroup_float8x8 Dm;
            simdgroup_load(Dm, Dg[sgitg], 8, 0, false);
#pragma unroll
            for (uint j = 0; j < 16; j++) { simdgroup_float8x8 t; simdgroup_multiply(t, Dm, O[j]); O[j] = t; }
        }
        simdgroup_half8x8 P0, P1;
        simdgroup_load(P0, Ps[sgitg], 16, 0, false);
        simdgroup_load(P1, Ps[sgitg] + 8, 16, 0, false);
#pragma unroll
        for (uint j = 0; j < 16; j++) {
            simdgroup_half8x8 V0, V1;
            simdgroup_load(V0, Vs + dh * 128 + j * 8, D, 0, false);
            simdgroup_load(V1, Vs + 8 * D + dh * 128 + j * 8, D, 0, false);
            simdgroup_multiply_accumulate(O[j], P0, V0, O[j]);
            simdgroup_multiply_accumulate(O[j], P1, V1, O[j]);
        }
    }

    /* normalize; each 8x8 output tile goes through this simdgroup's Dg slot
     * to the scattered (token, head) rows, gated by sigmoid(gate) */
    const float inv = l_row > 0.0f ? 1.0f / l_row : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tiisg; i < 64; i += 32) Dg[sgitg][i] = (i >> 3) == (i & 7u) ? simd_shuffle(inv, (ushort)((i >> 3) * 4)) : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Dm;
    simdgroup_load(Dm, Dg[sgitg], 8, 0, false);
#pragma unroll
    for (uint j = 0; j < 16; j++) {
        simdgroup_float8x8 t;
        simdgroup_multiply(t, Dm, O[j]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(t, Dg[sgitg], 8, 0, false);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tiisg; e < 64; e += 32) {
            const uint r = rt * 8 + (e >> 3);
            if (r >= nrows) continue;
            const uint64_t o = ((uint64_t)(tok0 + r / group) * H + kvh * group + r % group) * D +
                               dh * 128 + j * 8 + (e & 7u);
            out[o] = Dg[sgitg][e] * qwen4_sigmoid(gate[o]);
        }
    }
}

/* ---------------------------------------------------------------------------
 * GEMM search family (prefill projections).  y[tok][out] = x[tok][:] . W[out][:]
 * with a BM (weight rows) x BN (tokens) tile per threadgroup of WM x WN
 * simdgroups, K in steps of 32 staged in threadgroup memory as 8x8 tiles
 * (the kernel_mul_mm layout).  AQ: weights are PQ2_0 rows decoded while
 * staging, else F16 rows [out][in].  BH: activations are F16, else F32
 * (clamped to the fp16 range and rounded while staging).  PF: global loads
 * for the next K step are issued before the current step's math.  Every
 * variant multiplies the same half values in the same K order with float
 * accumulators, so all of them and upstream's kernel_mul_mm_f16_f32 on the
 * unpacked weight give bit-identical results.
 * ------------------------------------------------------------------------- */
struct bonsai2_u8 { half4 lo, hi; };

template <short BM, short BN, short WM, short WN, bool AQ, bool BH, bool PF>
kernel void kernel_bonsai2_gemm(
        constant bonsai2_mm_args &a [[buffer(0)]],
        device const uchar       *w [[buffer(1)]],
        device const uchar       *x [[buffer(2)]],
        device float             *y [[buffer(3)]],
        threadgroup char     *shmem [[threadgroup(0)]],
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr short BK = 32, NT = 32 * WM * WN, TM = BM / WM / 8, TN = BN / WN / 8;
    constexpr short AU = BM * 4, BU = BN * 4;              /* 8-wide K units per tile */
    constexpr short AUT = (AU + NT - 1) / NT, BUT = (BU + NT - 1) / NT;
    threadgroup half *sa = (threadgroup half *)shmem;      /* BM x BK */
    threadgroup half *sb = sa + BM * BK;                   /* BN x BK */

    const uint r0 = tg.y * BM, r1 = tg.x * BN;
    const short nr0 = (short)min(a.out_dim - r0, (uint)BM);
    const short nr1 = (short)min(a.n_tok - r1, (uint)BN);

    bonsai2_u8 ra[AUT], rb[BUT];

    /* A units: when they divide evenly, a thread takes AUT consecutive units
     * (adjacent 8-weight groups of one row, one scale load); otherwise unit
     * tid + u * NT.  Source pointers are fixed for the whole K loop; load(k)
     * only adds the K offset.  Past-the-edge rows read a valid row whose
     * results land outside the stored region. */
    constexpr bool AC = AU % NT == 0;
    auto aunit = [&](short u) -> short { return AC ? (short)(tid * AUT + u) : (short)(tid + u * NT); };
    device const uchar *pa[AUT];
    device const uchar *pb[BUT];
    short ga[AUT];
    for (short u = 0; u < AUT; u++) {
        const short unit = min(aunit(u), (short)(AU - 1));
        const short row = min((short)(unit / 4), (short)(nr0 - 1));
        ga[u] = unit % 4;
        pa[u] = AQ ? w + (ulong)(r0 + row) * a.row_bytes
                   : w + ((ulong)(r0 + row) * a.in_dim + 8 * ga[u]) * 2;
    }
    for (short u = 0; u < BUT; u++) {
        const short unit = min((short)(tid + u * NT), (short)(BU - 1));
        const short tok = min((short)(unit / 4), (short)(nr1 - 1)), g = unit % 4;
        pb[u] = x + ((ulong)(r1 + tok) * a.in_dim + 8 * g) * (BH ? 2 : 4);
    }

    auto load = [&](uint k) {
        for (short u = 0; u < AUT; u++) {
            if (!AC && aunit(u) >= AU) break;
            if (AQ) {
                device const uchar *blk = pa[u] + (k / BONSAI2_QK) * BONSAI2_QBYTES;
                const half d = *(device const half *)blk;
                const ushort q = *(device const ushort *)(blk + 2 + 2 * ((k % BONSAI2_QK) / 8 + ga[u]));
                for (short i = 0; i < 4; i++) {
                    ra[u].lo[i] = (half)((short)((q >> (2 * i)) & 3) - 1) * d;
                    ra[u].hi[i] = (half)((short)((q >> (2 * (i + 4))) & 3) - 1) * d;
                }
            } else {
                device const half4 *p = (device const half4 *)(pa[u] + (ulong)k * 2);
                ra[u].lo = p[0];
                ra[u].hi = p[1];
            }
        }
        for (short u = 0; u < BUT; u++) {
            if (BU % NT != 0 && tid + u * NT >= BU) break;
            if (BH) {
                device const half4 *p = (device const half4 *)(pb[u] + (ulong)k * 2);
                rb[u].lo = p[0];
                rb[u].hi = p[1];
            } else {
                device const float4 *p = (device const float4 *)(pb[u] + (ulong)k * 4);
                rb[u].lo = (half4)clamp(p[0], -65504.0f, 65504.0f);
                rb[u].hi = (half4)clamp(p[1], -65504.0f, 65504.0f);
            }
        }
    };
    auto store = [&]() {
        for (short u = 0; u < AUT; u++) {
            const short unit = aunit(u);
            if (!AC && unit >= AU) break;
            const short row = unit / 4, g = unit % 4;
            threadgroup half *dst = sa + 64 * (g * (BM / 8) + row / 8) + row % 8;
            for (short i = 0; i < 4; i++) {
                *(dst + 8 * i) = ra[u].lo[i];
                *(dst + 8 * (i + 4)) = ra[u].hi[i];
            }
        }
        for (short u = 0; u < BUT; u++) {
            const short unit = tid + u * NT;
            if (BU % NT != 0 && unit >= BU) break;
            const short tok = unit / 4, g = unit % 4;
            threadgroup half4 *dst = (threadgroup half4 *)(sb + 64 * (g * (BN / 8) + tok / 8) + 8 * (tok % 8));
            dst[0] = rb[u].lo;
            dst[1] = rb[u].hi;
        }
    };

    const short sm = sg % WM, sn = sg / WM;
    simdgroup_half8x8 ma[TM], mb[TN];
    simdgroup_float8x8 mc[TM * TN];
    for (short i = 0; i < TM * TN; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    if (PF) load(0);
    for (uint k = 0; k < a.in_dim; k += BK) {
        if (!PF) load(k);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        store();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (PF && k + BK < a.in_dim) load(k + BK);
        threadgroup const half *la = sa + 64 * (sm * TM);
        threadgroup const half *lb = sb + 64 * (sn * TN);
        for (short ik = 0; ik < BK / 8; ik++) {
            for (short i = 0; i < TM; i++) simdgroup_load(ma[i], la + 64 * i, 8, 0, false);
            for (short j = 0; j < TN; j++) simdgroup_load(mb[j], lb + 64 * j, 8, 0, false);
            for (short j = 0; j < TN; j++)
                for (short i = 0; i < TM; i++)
                    simdgroup_multiply_accumulate(mc[j * TM + i], mb[j], ma[i], mc[j * TM + i]);
            la += 64 * (BM / 8);
            lb += 64 * (BN / 8);
        }
    }

    const uint orow = r0 + 8 * TM * sm, otok = r1 + 8 * TN * sn;
    if (nr0 == BM && nr1 == BN) {
        device float *out = y + (ulong)otok * a.out_dim + orow;
        for (short j = 0; j < TN; j++)
            for (short i = 0; i < TM; i++)
                simdgroup_store(mc[j * TM + i], out + 8 * i + 8 * (ulong)a.out_dim * j, a.out_dim, 0, false);
        return;
    }
    /* Edge tile: each 8x8 accumulator through this simdgroup's 64-float slot. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *st = (threadgroup float *)shmem + 64 * sg;
    for (short j = 0; j < TN; j++) {
        for (short i = 0; i < TM; i++) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(mc[j * TM + i], st, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (short e = lane; e < 64; e += 32) {
                const short t = 8 * TN * sn + 8 * j + e / 8, o = 8 * TM * sm + 8 * i + e % 8;
                if (t < nr1 && o < nr0) y[(ulong)(r1 + t) * a.out_dim + r0 + o] = st[e];
            }
        }
    }
}

#define BONSAI2_GEMM_INST(BM_, BN_, WM_, WN_, AQ_, BH_, PF_, NAME_) \
template [[host_name("kernel_bonsai2_gemm_" #NAME_)]] \
kernel void kernel_bonsai2_gemm<BM_, BN_, WM_, WN_, AQ_, BH_, PF_>(constant bonsai2_mm_args &, \
        device const uchar *, device const uchar *, device float *, threadgroup char *, \
        uint2, ushort, ushort, ushort);
#define BONSAI2_GEMM_TILE(BM_, BN_, WM_, WN_) \
    BONSAI2_GEMM_INST(BM_, BN_, WM_, WN_, true,  false, false, BM_##x##BN_##_##WM_##x##WN_##_qfn) \
    BONSAI2_GEMM_INST(BM_, BN_, WM_, WN_, true,  false, true,  BM_##x##BN_##_##WM_##x##WN_##_qfp) \
    BONSAI2_GEMM_INST(BM_, BN_, WM_, WN_, true,  true,  false, BM_##x##BN_##_##WM_##x##WN_##_qhn) \
    BONSAI2_GEMM_INST(BM_, BN_, WM_, WN_, true,  true,  true,  BM_##x##BN_##_##WM_##x##WN_##_qhp) \
    BONSAI2_GEMM_INST(BM_, BN_, WM_, WN_, false, false, false, BM_##x##BN_##_##WM_##x##WN_##_ffn) \
    BONSAI2_GEMM_INST(BM_, BN_, WM_, WN_, false, false, true,  BM_##x##BN_##_##WM_##x##WN_##_ffp) \
    BONSAI2_GEMM_INST(BM_, BN_, WM_, WN_, false, true,  false, BM_##x##BN_##_##WM_##x##WN_##_fhn) \
    BONSAI2_GEMM_INST(BM_, BN_, WM_, WN_, false, true,  true,  BM_##x##BN_##_##WM_##x##WN_##_fhp)
BONSAI2_GEMM_TILE(64, 32, 2, 2)
BONSAI2_GEMM_TILE(64, 64, 2, 2)
BONSAI2_GEMM_TILE(64, 64, 2, 4)
BONSAI2_GEMM_TILE(128, 32, 4, 2)
BONSAI2_GEMM_TILE(128, 64, 4, 2)
BONSAI2_GEMM_TILE(64, 128, 2, 4)
BONSAI2_GEMM_TILE(128, 64, 4, 4)
BONSAI2_GEMM_TILE(128, 128, 4, 4)
BONSAI2_GEMM_TILE(32, 64, 1, 2)

/* Simdgroup MMA peak probe (tests/bonsai2/test_kernels --mma-peak): the
 * ceiling the prefill GEMMs can reach on this GPU.  Per simdgroup, each
 * iteration rewrites one element of its A tile, reloads A and runs NC
 * independent MMAs against NC different B tiles; every chain is stored, so
 * the compiler can neither hoist a product nor drop a chain. */
template <short NC>
kernel void kernel_bonsai2_mma_peak(
        constant bonsai2_mm_args &a [[buffer(0)]],
        device float *y [[buffer(1)]],
        uint tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half T[4][64];
    threadgroup half U[16 * 64];
    for (short i = lane; i < 64; i += 32) T[sg][i] = (half)(0.001f * (float)((i * 7 + tg) % 13));
    for (short i = tid; i < 16 * 64; i += 128) U[i] = (half)(0.002f * (float)((i * 3 + 1) % 11));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 A, B[NC];
    for (short i = 0; i < NC; i++) simdgroup_load(B[i], U + 64 * i, 8, 0, false);
    simdgroup_float8x8 C[NC];
    for (short i = 0; i < NC; i++) C[i] = make_filled_simdgroup_matrix<float, 8>((float)i);
    for (uint it = 0; it < a.in_dim; it++) {
        if (lane == 0) T[sg][it & 63] = (half)(float)(it & 7);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(A, T[sg], 8, 0, false);
        for (short i = 0; i < NC; i++) simdgroup_multiply_accumulate(C[i], A, B[i], C[i]);
    }
    for (short i = 0; i < NC; i++) simdgroup_store(C[i], y + ((tg * 4 + sg) * NC + i) * 64, 8, 0, false);
}
template [[host_name("kernel_bonsai2_mma_peak")]] kernel void kernel_bonsai2_mma_peak<8>(constant bonsai2_mm_args &, device float *, uint, ushort, ushort, ushort);
template [[host_name("kernel_bonsai2_mma_peak16")]] kernel void kernel_bonsai2_mma_peak<16>(constant bonsai2_mm_args &, device float *, uint, ushort, ushort, ushort);
