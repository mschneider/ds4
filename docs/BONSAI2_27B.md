# Ternary Bonsai 2 27B (fork)

[README](../README.md) · [fork maintenance](FORK_MAINTAINING.md)

This fork runs PrismML's [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
(`PQ2_0` GGUF, 7.2 GB) on the Metal backend. It is not part of upstream ds4.

## Quick start

```sh
make
./download_bonsai2.sh
./ds4 -m gguf/ternary-bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf -p "Hello"
./ds4-server -m gguf/ternary-bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf --ctx 32768
```

The server lists the model as `ternary-bonsai-2-27b`. It also accepts the
`qwen3.8-flash-next*` aliases, which select thinking or no-thinking. Chat
template, reasoning effort, tool calls, prefix caching and the KV disk cache
all behave as they do for Qwen3.8 Flash Next.

## Supported / not supported

| | |
|---|---|
| Backend | Metal only (Apple Silicon). Memory is 6.7 GiB of weights plus context buffers: about 10.3 GiB total at 4K context, 13.6 GiB at 8K. |
| Weights | `Ternary-Bonsai-2-27B-PQ2_0.gguf` only. PTQ1_0 and F16 are refused. |
| CLI, server, agent, bench | yes |
| Prefix cache, KV disk cache | yes |
| Batched multi-session decode | yes (`--batched-session N`): projections and head run once over all sessions; greedy output is identical to single-session runs. About 17 t/s aggregate at 4 sessions on the machine below. |
| Vision | yes, with `--vision gguf/ternary-bonsai-2-27b/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (Qwen3-VL tower shared with Qwen3.8; `./download_bonsai2.sh --vision`) |
| MTP | no (the GGUF has none) |
| CUDA / ROCm / CPU | no; the engine refuses at open |

## Model

Ternary Bonsai 2 is a ternary QAT of Qwen3.8-27B (GGUF arch `qwen35`). It has:

- 64 layers in the pattern 3 × Gated DeltaNet + 1 × gated GQA, with a 5120
  hidden size and dense SwiGLU FFNs (17408).
- GQA attention with 24 query and 4 KV heads, head dimension 256. q/k are
  normed, partial rope covers 64 dimensions (theta 1e7), and the output is
  gated with a sigmoid.
- GDN with 16 k and 48 v heads of 128, using the tiled V-head order.
- PQ2_0 linear weights: 128-weight blocks of one fp16 scale plus 2-bit
  codes, value = (code − 1)·d.
- A blockwise Hadamard rotation folded into every PQ2_0 weight:
  `y = W · H₁₀₂₄(s ⊙ P x)`. The ±1 sign vector `s` depends on the input
  width (5120 / 6144 / 17408). `P` is a head permutation, used only at the
  GDN `ssm_out` input. The token embedding stores the latent row `z`, and
  `h = s ⊙ H₁₀₂₄(z)`. Image embeddings come from the mmproj already in the
  primal basis: ds4 rotates them forward on the host, so that the batch's
  shared embedding unrotate leaves them unchanged. The GGUF carries the contract in
  `prism.hadamard.*`, and ds4 validates every key and refuses files that
  differ.

ds4 runs Bonsai as a variant (`DS4_VARIANT_BONSAI2`) of the Qwen3.8 family. It
shares that family's sessions, state payloads, tokenizer, chat template, the
GDN conv/prep/scan kernels and the attention decode kernels. Only the forward
graph and these kernels are Bonsai's own:

| Kernel (`metal/bonsai2.metal`) | Role |
|---|---|
| `mv_pq2` | PQ2_0 matvec (1–8 tokens) |
| `gemm_*` | Prefill GEMM family (9+ tokens): tile shape × PQ2_0 or F16 weights × F32 or F16 activations × prefetch, all bit-identical; see "Prefill GEMM" |
| `unpack_pq2_f16` | PQ2_0 → F16 into a scratch matrix for the F16-weight GEMMs (and upstream's `kernel_mul_mm_f16_f32`) |
| `mma_peak` | Simdgroup MMA throughput probe (`test_kernels --mma-peak`) |
| `rms_rotate` | RMSNorm, then sign and 1024-FWHT (input of every projection) |
| `gdn_out_rotate` | GDN head norm with a **silu(z)** gate (Qwen3.8 Flash Next uses sigmoid), tiled→grouped permutation, rotation |
| `swiglu_rotate`, `rotate`, `unrotate` | FFN down input, attention output input, embedding inverse |
| `gemv_bf16_pair` | GDN alpha/beta BF16 projections while decoding |
| `attn_prep` | q/k norm + rope + gate split without the Qwen3.8 Flash Next indexer |
| `attn_fa` | Prefill attention (9+ tokens), flash style: one threadgroup per (KV head, block of 4 query tokens) shares each staged K/V tile among all 24 (token, head) rows; decode keeps the Qwen3.8 kernels |

## Verification

```sh
make test-bonsai2-kernels      # model-free kernel checks vs double-precision CPU references,
                               # plus every prefill GEMM configuration bit-exact (--gemm-check)
make test-bonsai2-reference    # greedy tokens + top-5 logprobs vs PrismML's llama.cpp fork
```

`tests/bonsai2/golden.json` holds six chat cases: five short ones and a
3960-token Italian prefill. They were generated greedily (128 tokens) with
PrismML's fork `prism@11809184` on the pinned GGUF. ds4 reproduces every
generated token. The top-5 logprob drift is ≤ 0.012 (gate 0.05).

## Performance

### Prefill GEMM

Projections are about 85% of prefill time (per-op profile of a 4069-token
prompt: FFN gate/up/down 60%, the other projections 25%, attention 9%, GDN
scan 1.6%, everything else about 4%). They run on `kernel_bonsai2_gemm`, a
family of simdgroup GEMMs with four axes:

- **tile**: BM weight rows × BN tokens per threadgroup on WM × WN
  simdgroups, K in steps of 32 staged as 8×8 tiles (`64x32_2x2`,
  `64x64_2x2`, `64x64_2x4`, `128x32_4x2`, `128x64_4x2`, `64x128_2x4`,
  `128x64_4x4`, `128x128_4x4`, `32x64_1x2`);
- **weights**: PQ2_0 decoded while staging (`q`), or F16 rows unpacked
  once per call (`f`);
- **activations**: F32 rounded to half while staging (`f`), or F16 (`h`),
  which the rotation kernels then write directly;
- **prefetch**: global loads for the next K step issued before the current
  step's math (`p`) or not (`n`).

All 72 configurations multiply the same half values in the same K order with
float accumulators, so their outputs are bit-identical to each other and to
upstream's `kernel_mul_mm_f16_f32` on the unpacked weight (`make
test-bonsai2-kernels` checks this exactly). Choosing among them is purely a
speed question. Writing the activations as half does not change results
either: the rotations store exactly the value the GEMM would round to
(clamped to the fp16 range).

Defaults (M4 Pro): **half activations** for every prefill GEMM, PQ2_0
weights below 2048 tokens and **F16 weights from 2048**, **no prefetch**, and
an automatic tile choice between `64x32_2x2` and `64x64_2x2`: the wide tile
costs about 0.91× the narrow one per padded token row, so the narrow one
runs only where padding to 64 wastes more (9–32, 65–96, 129–160, 193–224
and 257–288 tokens).

Each thread stages consecutive 8-weight groups of one weight row (one block
scale read for 16 weights in the narrow tile) from source pointers computed
once per threadgroup. That loader layout was worth 2–4% on the narrow tile
and about 1% on the wide one over per-step address math with units 32 rows
apart, and it fixed a 2% loss against the old fused kernel at 219 tokens.

How close that is to the hardware: `test_kernels --mma-peak` measures the
simdgroup MMA ceiling of the M4 Pro at 6.0–6.2 TFLOPS. The best
configurations reach 5.8–5.9 TFLOPS on every Bonsai shape from 512 tokens on,
about 95% of it, so further GEMM tuning on this chip has little left.

Kernel grid (`make tune-bonsai2-gemm`; whole-model GEMM time weighted by how
often each shape runs; percentages against the previous production path,
the fused 64×32 GEMM below 512 tokens and unpack plus upstream's GEMM from
512):

| tokens | best configuration | gain |
|---:|---|---:|
| 16–32 | `64x32_2x2_qh*` | equal (same tile as before) |
| 64 | `64x64_2x2_qhn` | +6.4% |
| 128 | `64x64_2x2_qhn` | +7.7% |
| 256 | `64x64_2x2_qhn` | +8.3% |
| 512 | `64x64_2x2_qhp` / `qhn` | +5.9% |
| 1024 | `64x64_2x2_qhn` | +2.6% |
| 2048 | `64x64_2x2_fhn` | +1.8% |
| 4096 | `32x64_1x2_fhn` / `64x64_2x2_fhn` | +1.9% |
| 8192 | `32x64_1x2_fhn` | +1.1% |

Prefetch never helped, larger tiles (128 rows or tokens, 8–16 simdgroups)
lost on every shape, and F16 weights pay for their unpack only from about
2048 tokens.

Whole-model prefill (t/s), M4 Pro on AC in High Power Mode, cold process
per run, median of 2–3 interleaved runs, same greedy output everywhere:
`5c1523e` (same attention, previous fused 64×32 GEMM), `b5ad4de` (first
GEMM family), and the current loader with auto ratio 0.91:

| prompt | `5c1523e` | `b5ad4de` | current |
|---|---:|---:|---:|
| 169 tokens | 86.7 | 94.1 | 95.2 |
| 219 tokens | 96.6 | 95.3 | 97.2 |
| 279 tokens | | 95.5 | 97.8 (ratio 0.88: 97.0) |
| 319 tokens | 99.9 | 108.5 | 109.9 |
| 569 tokens | 102.8 | 109.5 | 110.7 |
| 4069 tokens | 108.0 | 110.0 | 111.8 |
| ~16k tokens | 100.7 | 103.5 | 104.5 |

The `5c1523e` column for 4069 and ~16k tokens comes from an earlier grid
session; everything else is from same-session interleaved runs. Short
prompts carry a one-time cost of about 2.5 t/s for pipeline creation inside
the first prefill, the same for every version.

In the same grid, half activations off cost about 3% at every length,
unpacking from 1024 instead of 2048 tokens was within noise, unpacking never
cost about 1% at 4069 tokens, and prefill chunks of 4096 and 8192 (the
default) tied while 2048 was slower. Qwen-family prefill chunks come from
`DS4_QWEN4_PREFILL_CHUNK`; `--prefill-chunk` does not apply.

### Tuning on other Macs

The defaults above are measured on one M4 Pro. Every choice is a runtime
setting, and none of them changes results (all configurations are
bit-identical), so tuning on another chip is safe:

| Variable | Default | Meaning |
|---|---|---|
| `DS4_BONSAI2_GEMM_TILE` | `auto` | force one tile, e.g. `64x64_2x2` |
| `DS4_BONSAI2_GEMM_AUTO` | `64x32_2x2,64x64_2x2,0.91` | auto's narrow tile, wide tile, and the wide tile's cost per padded token relative to the narrow one |
| `DS4_BONSAI2_GEMM_PREFETCH` | `0` | prefetch the next K step |
| `DS4_BONSAI2_UNPACK_MIN` | `2048` | tokens from which weights are unpacked to F16 (`off`: never) |
| `DS4_BONSAI2_HALF_ACT_MIN` | `9` | tokens from which GEMM inputs are half (`off`: never) |
| `DS4_BONSAI2_UPSTREAM_MIN` | `off` | tokens from which upstream's `kernel_mul_mm_f16_f32` runs on the unpacked weight instead of the family (Metal 4 tensor ops on M5 and later; activations then stay F32) |
| `DS4_QWEN4_PREFILL_CHUNK` | `8192` | prefill chunk size (the token count every GEMM sees) |

Setting any `DS4_BONSAI2_*` variable prints the effective configuration
once, including whether tensor ops are available.

Workflow:

1. `tests/bonsai2/test_kernels --mma-peak`: the GPU's simdgroup MMA ceiling.
2. `make tune-bonsai2-gemm` (`GEMM_TOKENS=...`, `GEMM_FILTER=...` narrow
   it): checks every configuration bit-exact, times all of them, the
   production paths (`prod_fused`, `prod_unpack`, `prod_upstream`) on the
   seven Bonsai projection shapes, and ranks them with
   `tests/bonsai2/gemm_rank.py`, which also prints the narrow/wide
   per-padded-token ratio for `DS4_BONSAI2_GEMM_AUTO`.
3. Translate the ranking into settings: the best tile pair and ratio, the
   token count where `f` overtakes `q` (`DS4_BONSAI2_UNPACK_MIN`), where
   `prod_upstream` overtakes the family (`DS4_BONSAI2_UPSTREAM_MIN`).
4. Confirm end to end, since kernels interact through caches and chunk
   sizes: `tests/bonsai2/prefill_grid.sh "X=1" "DS4_BONSAI2_UNPACK_MIN=1024"
   ...` runs the whole model on several prompt lengths per configuration
   (AC and High Power Mode required), and every configuration must print the
   same greedy text.
5. `make test-bonsai2-reference` with the chosen settings exported.

What to expect per generation:

- **M1–M4** have no Metal 4 tensor ops; every GEMM runs on simdgroup MMAs,
  as on the M4 Pro here. Expect the same structure of results: half
  activations a win everywhere (they halve activation traffic), 64-row tiles
  with 32/64-token widths best, prefetch neutral. What likely moves: the
  auto ratio (core count and occupancy), and the unpack threshold, which
  trades an extra pass over up to 178 MB of F16 per projection against
  cheaper inner loops: lower-bandwidth chips (M1/M2 base, 68–100 GB/s)
  should unpack later or never, Max chips earlier. M3 and M4 allocate
  registers and threadgroup memory dynamically, M1 and M2 do not, so tile
  rankings can differ there. With 16 GB of memory, smaller prefill chunks
  keep the per-chunk buffers small.
- **M5** adds tensor ops ("neural accelerators") in each GPU core, and
  upstream's F16 GEMM uses them. Expect `prod_upstream` to beat the
  simdgroup family by a wide margin from a few dozen tokens on, so the main
  setting is `DS4_BONSAI2_UPSTREAM_MIN` (try 32 to 512); below it the
  family with half activations still runs. With much faster GEMMs, the
  per-call unpack (bandwidth bound) and the parts that stay on simdgroup
  MMAs (prefill attention, GDN scan, rotations) become a larger share, so
  re-profile, and larger prefill chunks amortize the unpack better.
  `attn_fa`'s shape (24 rows, K then V through 8 KB) is fixed at compile
  time; on M5 a tensor-op attention kernel is the next candidate.

### Findings to verify on another machine

Each line is a claim from the M4 Pro and the command that checks it.

| Claim (M4 Pro) | Check |
|---|---|
| All 72 GEMM configurations and upstream's F16 GEMM agree bit for bit | `tests/bonsai2/test_kernels --gemm-check` |
| Half activations change no result | `make test-bonsai2-kernels` (`half activations ...` lines) |
| `attn_fa` equals the Qwen3.8 attention kernel bit for bit | `make test-bonsai2-kernels` (`attn_fa ... vs qwen4 kernel`) |
| Reference drift unchanged: top-5 logprob diffs 0.0118 / 0.0034 / 0.0024 / 0.0030 / 0.0028 / 0.0102 | `make test-bonsai2-reference` |
| Simdgroup MMA ceiling 6.0–6.2 TFLOPS | `test_kernels --mma-peak` |
| Best GEMMs at ~95% of that ceiling from 512 tokens | `make tune-bonsai2-gemm` |
| `attn_fa` 3.3–3.4 TFLOPS at 4–8k tokens, ~3.8× the Qwen3.8 kernel | `test_kernels --bench-attn` |
| Prefill table above (defaults vs `5c1523e`) | `tests/bonsai2/prefill_grid.sh` with both binaries |
| Half activations off cost ~3% end to end | `prefill_grid.sh "X=1" "DS4_BONSAI2_HALF_ACT_MIN=off"` |
| Auto ratio 0.91 (narrow tile at 9–32, 65–96, 129–160, 193–224, 257–288 tokens) | `gemm_rank.py` ratio line from 96 tokens on; `prefill_grid.sh -p 21 "X=1" "DS4_BONSAI2_GEMM_AUTO=64x32_2x2,64x64_2x2,0.88"` |

### Prefill attention

Bonsai's 16 full-attention layers prefill with `attn_fa` instead of the
Qwen3.8 kernel, which ran one threadgroup per (KV head, token): every token
re-read the whole K/V history, and its 6 query heads were padded to 16
matrix rows. The flash kernel stages each K/V tile once for 4 tokens x 6
heads = 24 rows and keeps the same inner loop, so its output is
bit-identical (the kernel tests compare against the old kernel exactly).
One layer, M4 Pro (`tests/bonsai2/test_kernels --bench-attn`):

| tokens | Qwen3.8 kernel | `attn_fa` |
|---:|---:|---:|
| 512 | 4.67 ms (0.69 TFLOPS) | 1.54 ms (2.10) |
| 2048 | 61.6 ms (0.84) | 16.4 ms (3.14) |
| 4096 | 237 ms (0.87) | 62.9 ms (3.28) |
| 8192 | 913 ms (0.90) | 242 ms (3.41) |

The shape came from a sweep at 4096 tokens (TFLOPS): queries in registers
instead of threadgroup memory 1.5 (register pressure), 48 rows 1.5–1.6, K
and V staged together 1.6, 16 rows 2.4, 32 rows 2.9, 24 rows with K then V
through one 8 KB buffer 3.3. Addressing that buffer through pointer aliases
instead of the array itself is worth 3.3 vs 2.95 (Metal codegen; the kernel
says so where it matters).

Whole-model prefill, default chunk, interleaved runs, same greedy output:
4069 tokens about 100 → 107.8 t/s, about 16k tokens 79.8 → 101.2 t/s.

### Against PrismML llama.cpp

M4 Pro (16-core GPU, 48 GB) with macOS Low Power Mode on, measured back to
back with the reference fork on the same machine (`llama-bench -fa 1`),
before the unpack path existed:

| | ds4 | PrismML llama.cpp |
|---|---:|---:|
| prefill 512 | 50–62 t/s | 32.6 t/s |
| decode | 11.5–13.3 t/s | 7.5 t/s |

The PQ2_0 matvec decodes codes as fp16 denormals (one AND per two weights;
see `metal/bonsai2.metal`) and streams at about the copy rate in short bursts:
0.10 ms for 5120 → 17408, 230 GB/s (`make bench-bonsai2-kernels`). Under
Low Power Mode the GPU clock drops after a burst. The same kernel run 3000
times back to back takes 0.23 ms per call, and in the model it averages
0.17 ms. Decode numbers therefore depend on power state and thermals more
than on the kernels. Re-measure with Low Power Mode off before quoting
absolute figures.
