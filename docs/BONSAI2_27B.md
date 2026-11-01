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
| `mv_pq2` / `mm_pq2` | PQ2_0 matvec (1–8 tokens) and simdgroup GEMM (9–511 tokens) |
| `unpack_pq2_f16` | PQ2_0 → F16 into a scratch matrix, then upstream's `kernel_mul_mm_f16_f32` (512+ tokens) |
| `rms_rotate` | RMSNorm, then sign and 1024-FWHT (input of every projection) |
| `gdn_out_rotate` | GDN head norm with a **silu(z)** gate (Qwen3.8 Flash Next uses sigmoid), tiled→grouped permutation, rotation |
| `swiglu_rotate`, `rotate`, `unrotate` | FFN down input, attention output input, embedding inverse |
| `gemv_bf16_pair` | GDN alpha/beta BF16 projections while decoding |
| `attn_prep` | q/k norm + rope + gate split without the Qwen3.8 Flash Next indexer |
| `attn_fa` | Prefill attention (9+ tokens), flash style: one threadgroup per (KV head, block of 4 query tokens) shares each staged K/V tile among all 24 (token, head) rows; decode keeps the Qwen3.8 kernels |

## Verification

```sh
make test-bonsai2-kernels      # model-free kernel checks vs double-precision CPU references
make test-bonsai2-reference    # greedy tokens + top-5 logprobs vs PrismML's llama.cpp fork
```

`tests/bonsai2/golden.json` holds six chat cases: five short ones and a
3960-token Italian prefill. They were generated greedily (128 tokens) with
PrismML's fork `prism@11809184` on the pinned GGUF. ds4 reproduces every
generated token. The top-5 logprob drift is ≤ 0.012 (gate 0.05).

## Performance

### Prefill GEMM: fused or unpacked

From 512 tokens on, each projection unpacks its PQ2_0 weight to F16 once and
runs upstream's F16 GEMM instead of the fused PQ2_0 GEMM. The unpack is one
extra pass over a matrix of up to 178 MB, and upstream's GEMM is faster at
large token counts. Whole-model prefill of a 4069-token prompt on the M4 Pro
below, in High Power Mode on AC, interleaved runs, same greedy output:

| prefill chunk (`DS4_QWEN4_PREFILL_CHUNK`) | fused | unpack + F16 |
|---|---:|---:|
| 512 | 94.0 t/s | 96.1 t/s |
| 1024 | 94.0 t/s | 98.9 t/s |
| 2048 | 94.1 t/s | 100.3 t/s |
| 4096 (default 8192 behaves the same) | 94.1 t/s | 100.9 t/s |

Shorter prompts lose with the unpack: 316 tokens 99.0 → 97.4 t/s, 186
tokens 95.8 → 88.7 t/s. Hence the 512-token threshold
(`ds4_gpu_bonsai2_set_unpack_min_tokens`). Note that Qwen-family prefill
chunks come from `DS4_QWEN4_PREFILL_CHUNK`; `--prefill-chunk` does not apply.

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
