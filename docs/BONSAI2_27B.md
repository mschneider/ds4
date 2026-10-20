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

Ternary Bonsai 2 is a ternary QAT of Qwen3.5-27B (GGUF arch `qwen35`). It has:

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
| `mv_pq2` / `mm_pq2` | PQ2_0 matvec (1–8 tokens) and simdgroup GEMM (prefill) |
| `rms_rotate` | RMSNorm, then sign and 1024-FWHT (input of every projection) |
| `gdn_out_rotate` | GDN head norm with a **silu(z)** gate (Qwen3.8 uses sigmoid), tiled→grouped permutation, rotation |
| `swiglu_rotate`, `rotate`, `unrotate` | FFN down input, attention output input, embedding inverse |
| `gemv_bf16_pair` | GDN alpha/beta BF16 projections while decoding |
| `attn_prep` | q/k norm + rope + gate split without the Qwen3.8 indexer |

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

M4 Pro (16-core GPU, 48 GB) with macOS Low Power Mode on, measured back to
back with the reference fork on the same machine (`llama-bench -fa 1`):

| | ds4 | PrismML llama.cpp |
|---|---:|---:|
| prefill 512 | 60–65 t/s | 32.6 t/s |
| decode | 11.3 t/s | 7.5 t/s |

Decode is limited by the PQ2_0 matvec. Its 2-bit decode is ALU-bound at
about 165 GB/s of weight streaming (`make bench-bonsai2-kernels`), against
roughly 245 GB/s for a plain read. Normal power mode should help both
numbers. Re-measure there before quoting absolute figures.
