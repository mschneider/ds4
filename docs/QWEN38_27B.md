# Qwen3.8 27B (fork, in progress)

[README](../README.md) · [fork maintenance](FORK_MAINTAINING.md) ·
[Bonsai 2 27B](BONSAI2_27B.md) · [Qwen3.8 Flash Next](QWEN38_FLASH_NEXT.md)

This branch (`qwen38-27b`) adds the dense
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) to the Metal
backend, with 4-bit weights and the model's built-in MTP head for speculative
decoding. It is not part of upstream ds4. The goal is the fastest local decode
on Apple Silicon at no measurable quality cost against a Q4_K_M reference.

## Status

| Step | State |
|---|---|
| 1. Load plain `qwen35` GGUFs as a variant beside Bonsai 2; one-token and prefill graph | done: greedy tokens match llama.cpp on all six reference cases |
| 2. Measure per-token overhead outside weight streaming | first run: 12.1 tok/s on 19.4 GB (≈235 GB/s), controlled bench pending |
| 3. Q4_K matvec that reads each weight once for 1–4 rows | kernel done and tested; timing pending |
| 4. MTP head on the existing Qwen3.8 speculative cycle | designed (below), todo |
| 5. All-4-bit GGUF recipe and KLD/quality check against Q4_K_M and BF16 | recipe tool done; imatrix and KLD todo |

Each step records what it measured here when it lands.

## Usage

```sh
make
python3 -m venv misc/qwen38-27b/venv && misc/qwen38-27b/venv/bin/pip install numpy huggingface_hub
make -C gguf-tools quants-shared
hf download ggml-org/Qwen3.8-27B-GGUF Qwen3.8-27B-BF16.gguf --local-dir gguf/qwen38-27b
misc/qwen38-27b/venv/bin/python gguf-tools/qwen38_27b_quantize.py \
    --src gguf/qwen38-27b/Qwen3.8-27B-BF16.gguf --out gguf/qwen38-27b/Qwen3.8-27B-Q4K.gguf
./ds4 -m gguf/qwen38-27b/Qwen3.8-27B-Q4K.gguf -c 32768
```

The server lists the model as `qwen3.8-27b`. Any `qwen35` GGUF whose
projections are Q4_K, Q8_0, BF16, F16 or F32 loads; ggml-org's Q4_K_M uses
Q6_K for `attn_output` and `output`, which `--keep-quantized --proj Q8_0`
converts while copying everything else.

## Model

Qwen3.8-27B is its own release (HF repo created 2026-08-05, public around
08-13), "built on the architectural foundation of Qwen3.5". It is a VLM
(`Qwen3_5ForConditionalGeneration`, HF `model_type` `qwen3_5`); the llama.cpp
GGUF architecture is `qwen35`, the same as Ternary Bonsai 2, which is a ternary
QAT of this model.

- 64 layers: 16 × (3 Gated DeltaNet + 1 gated GQA), hidden 5120, dense SwiGLU
  FFN 17408, vocabulary 248320, untied embedding and output head.
- GQA: 24 query and 4 KV heads of 256, q/k RMSNorm, 64-dimensional interleaved
  MRoPE (theta 1e7), sigmoid output gate.
- GDN: 16 key and 48 value heads of 128, conv width 4, output norm gated with
  silu(z) (Qwen3.8 Flash Next uses sigmoid).
- One MTP layer: `fc` (10240 → 5120) over the normed hidden state and the next
  token's normed embedding, then one full-attention block and FFN, sharing the
  main embedding and output head.
- 262144-token native context (1M with YaRN). Thinking is on by default.

There is no dense Qwen4. `qwen4exp` in this repository is the architecture
name of Qwen3.8 Flash Next (an MoE), which is why the shared Qwen code uses the
`qwen4` prefix.

### What ds4 already shares with this model

The geometry of the GDN and attention layers matches Qwen3.8 Flash Next, so the
27B runs as a variant of that family, as Bonsai 2 does. It reuses the family's
sessions, state payloads, tokenizer, chat template, GDN conv/prep/scan kernels
and attention decode kernels. Flash Next's MTP cycle already has everything
speculative decoding needs on a hybrid recurrent model:

- 2- and 3-row verify passes with row-batched GDN scan, conv and attention
  kernels (`kernel_qwen4_gdn_scan_rows2`, `kernel_qwen4_conv_stream_rows2`,
  `kernel_qwen4_attn_decode_rows`).
- Per-row recurrent state snapshots during verify; a rejection swaps a snapshot
  pointer instead of recomputing (`qwen4_graph_state_swap2`).
- An adaptive depth policy (`qwen4_spec_depth`): one draft by default, a second
  chained draft while recent first drafts are all accepted.
- Draft-head vocabulary trimming (`DS4_QWEN4_MTP_DRAFT_VOCAB`).

On Flash Next this gives +18% on prose and +65% on deterministic lists
(M3 Ultra, `speed-bench/qwen38-checkpoints`). A dense model should gain more,
because verifying two rows costs almost nothing extra when every step is bound
by streaming the same weights.

## Quantization landscape (2026-10-31)

Qwen ships only an FP8 checkpoint; every GGUF and MLX quant is community made.

| Source | What | Notes |
|---|---|---|
| [ggml-org](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) | Q4_K_M 18.97 GB, Q8_0 28.6 GB, BF16 53.8 GB; MTP Q4_0/Q8_0/BF16; DFlash drafts; mmproj | Reference files for this branch |
| [bartowski](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) | full K/IQ ladder | re-uploaded 2026-09-11 with a new tensor layout; publishes KLD (below) |
| [unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD dynamic quants 6.2–31.5 GB, separate MTP | re-quantized 08-19; NVFP4 variant |
| [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | 2.5–3.5 bpw | IQ3_S wikitext PPL 7.07 vs BF16 7.05 |
| [prism-ml Bonsai 2](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | ternary QAT, PQ2_0 7.2 GB | supported on `bonsai2` |
| [mlx-community](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | affine 4-bit g64 16.05 GB, mxfp4, nvfp4; drafter-only `-MTP-*` repos 0.24 GB | everything 4-bit, including `lm_head` and GDN `in_proj_a/b` |

bartowski's measurements against BF16 (wikitext):

| Quant | PPL | KLD | same top token |
|---|---:|---:|---:|
| Q8_0 | 6.75 | 0.0009 | 98.7% |
| Q6_K | 6.74 | 0.0030 | 97.4% |
| Q5_K_M | 6.75 | 0.0053 | 96.9% |
| Q4_K_M | 6.76 | 0.0139 | 95.0% |
| IQ4_XS | 6.78 | 0.0188 | 94.1% |
| IQ3_M | 6.92 | 0.041 | |
| IQ2_M | 7.53 | 0.149 | |

ggml-org's Q4_K_M keeps more than the FFN at higher precision: Q8_0 for
`attn_qkv`, `attn_gate`, `ssm_out`, `attn_q/k/v` and `ssm_alpha/beta`
(7.16 GB); Q6_K for `attn_output` and `output`; Q4_K only for the FFN
(9.63 GB) and `token_embd`. That costs 18.25 GB of weight reads per token
against 14.4 GB for MLX's all-4-bit file.

## How MLX and llama.cpp run MTP

| | MLX (mlx-lm #990, open) | llama.cpp (`--spec-type draft-mtp`, #22673) |
|---|---|---|
| Weight bytes per token | 14.4 GB, all 4-bit | 18.25 GB (Q4_K_M) |
| Draft depth | 1 (deeper on a branch; mlx-vlm and MTPLX use 3) | 3 by default |
| Verify T rows vs 1 | ~1.0 at T=2 after `qmv_wide` (mlx #3764); 1.38/1.79 at T=3/4 before it (M4 Pro) | 1.32/1.59/1.73 at T=2/3/4 (Q4_0, M3 Ultra, #29869); Q4_K at T=2 re-reads the weights, since its few-row path starts at T=3 |
| Draft cost | 0.24 GB head + 0.72 GB shared `lm_head` | same per step, plus a catch-up pass; the MTP sidecar duplicates `token_embd` and `output` (1.4 of 1.7 GB) |
| GDN rollback | snapshot after the confirmed row | per-row state written by the GDN kernel; rejection selects an index |
| Known problems | server memory growth (#1807, #1871); double RMSNorm shift on MTP checkpoints (#1623, fixed) | 15 GB host growth over 96K tokens on Metal (#30174); Vulkan regressions |

Measured with MTP (author-reported):

- M4 Pro, MLX, 27B 4-bit, one draft: 15.7 → 24.6 tok/s (1.57×), 88% accepted
  at temperature 0, 80–85% at 0.6–1.0.
- M4 Max, oMLX: 1.84× at 1K context, 2.18× at 4K.
- M5 Max, MTPLX depth 3: 2.24×.
- M3 Ultra, llama.cpp Q8_0: 21.9 → 34.2 tok/s with one draft (89% accepted);
  44.5 with two drafts and the multi-column Q8_0 matvec of #29110.
- Gains shrink with context: 17.1 → 20.3 tok/s at 89K (acceptance 0.68).
- MTP loses on bf16/fp16 weights, where verifying more rows is no longer free.

## Speed model and prediction

Decode streams about 25.8 B weights per token (the output head's 1.27 B
included; the embedding is a row gather), so at 4.5 bits per weight a step
reads about 14.5 GB. An M4 Pro (273 GB/s) sustains about 226 GB/s in MLX.

| M4 Pro (16-core GPU) | Prediction |
|---|---:|
| ordinary decode, 4-bit | ~15 tok/s |
| MTP, one draft (verify 2 rows ≈ 1.03×, draft ≈ 0.07×, 85% accepted) | ~25 tok/s |
| MTP, adaptive second draft (verify 3 rows ≈ 1.08×, second draft 60–70% accepted) | 28–30 tok/s |
| trimmed draft vocabulary | about +5% more |

For comparison: MLX measures 24.6 tok/s on the same chip; llama.cpp's
Q4_K_M with MTP should land around 15–18 tok/s.

The prediction rests on three requirements:

1. **Bytes.** Every large projection 4-bit, as in MLX. Keeping only the tiny
   GDN `ssm_alpha/beta` at Q8_0 or BF16 costs nothing measurable.
2. **A multi-row 4-bit matvec.** The existing `kernel_mul_mv_q4_K_dense_f32`
   (530–650 GB/s on M5 at one row) dispatches one threadgroup grid per token,
   so two rows read the weights twice. Verification needs a kernel that
   dequantizes each block once and applies it to 1–4 rows.
3. **Low fixed overhead.** Bonsai's 7.2 GB of weights would allow about
   30 tok/s at its measured matvec bandwidth, yet it decodes at 11.5–13.3
   tok/s (Low Power Mode on). The 27B graph must keep the rotations and the
   extra kernels out of the decode step, and step 2 measures how much time is
   spent outside weight streaming before any MTP work.

## Design

The 27B is a second fork variant of the Qwen3.8 family, next to Bonsai 2, and
shares Bonsai's `qwen35` loader, shape and graph skeleton. Bonsai's graph is
this model plus a Hadamard rotation in front of every projection, so the plain
variant drops the rotations and calls the ordinary projections instead. Fork
rules apply: code in fork-owned files, upstream files only get one-line hooks.

Weight formats accepted at first: Q4_K, Q8_0, BF16, F16 and F32 for
projections, which the Qwen3.8 code already multiplies. Q6_K (ggml-org's
`attn_output` and `output`) needs a kernel or a requantized file; the target
recipe below avoids it.

### Target GGUF recipe

All-4-bit from the BF16 GGUF, with the repository's quantizer library
(`make -C gguf-tools quants-shared`) and an imatrix:

| Tensors | Type |
|---|---|
| FFN gate/up/down, GDN `attn_qkv`/`attn_gate`/`ssm_out`, attention q/k/v/o, `output`, MTP projections | Q4_K |
| `token_embd` | Q4_K (row gather only) |
| `ssm_alpha`, `ssm_beta` (6.6 MB each) | BF16 |
| norms, `ssm_conv1d`, `ssm_a`, `ssm_dt` | F32 |

MTP lives in the main GGUF (`qwen35.nextn_predict_layers = 1`, layer 64) and
shares `token_embd` and `output`, so no sidecar duplicates them. Acceptance
must be checked with the head at Q4_K and at Q8_0; mlx-lm #990 found the
difference within ±0.3 points.

## Verification

`make test-qwen38-27b-reference` runs Bonsai's six reference cases (five
short chats and a 3960-token Italian passage) against llama.cpp's greedy
output on the same GGUF. ds4 reproduces every token. Top-5 logprobs differ by
up to 0.32, against 0.012 for Bonsai, and the difference is llama.cpp's:

- llama.cpp against itself (flash attention off, ubatch 16) differs by at most
  0.019, so the gap is systematic, not run-to-run noise.
- Per-layer hidden states agree to about 3e-4 relative at layer 0 and drift
  slowly through 64 layers; the first visible gap is layer 0's FFN.
- A float64 numpy recomputation of that FFN from the same weights (gguf-py
  dequantization) matches ds4 to 2e-7 relative and llama.cpp only to 3e-3:
  llama.cpp's few-row Q4_K matmuls round activations to half.

The golden file therefore checks tokens exactly and logprobs at 0.4.

## Validation plan

- Kernel tests against double-precision CPU references, model-free, in
  `tests/qwen38_27b/`, as Bonsai's `make test-bonsai2-kernels` does.
- Greedy tokens and top-5 logprobs against llama.cpp on the same GGUF
  (Bonsai's golden-file pattern, `tests/bonsai2/check_reference.py`).
- KLD and same-top-token against the BF16 GGUF for the recipe, compared with
  ggml-org Q4_K_M; `gguf-tools/quality-testing` for continuation scores.
- MTP: greedy output must be byte-identical with and without `--mtp`;
  acceptance and speed recorded per prompt type at 4K and 32K context.
- Speed: `ds4-bench` back to back with llama.cpp and MLX on the same machine,
  Low Power Mode off.

## Open questions

- (Settled) V-head order of plain GGUFs: llama.cpp's converter reorders
  `ssm_out`'s columns to the tiled order (`conversion/qwen.py`), so the plain
  graph gates the scan output in place with no permutation.
- Whether the Q4_K dequantization is cheap enough for 4 rows to stay
  bandwidth-bound on M4 Pro, or whether a simpler 4.5 bpw block format
  (Q4_0-like with a min) is faster at equal quality.
- Long-context behavior: attention traffic grows and acceptance falls; the
  adaptive depth policy may need a context term.

## MTP design (step 4)

From llama.cpp's `graph_mtp` (`src/models/qwen35.cpp`):

```
h   = output_norm(trunk residual of row t)                 (h_nextn)
x   = eh_proj([enorm(embd(next token)), hnorm(h)])         10240 -> 5120
x  += attention(attn_norm(x))                              layer 64's own KV cache
x  += ffn(post_attention_norm(x))
h'  = shared_head_norm(x)                                  chained drafts feed h' back
logits = output(h')
```

The GGUF stores it as `blk.64.*` with `nextn.{eh_proj, enorm, hnorm,
shared_head_norm}` and `block_count` 65. ds4's Qwen3.8 cycle calls
`qwen4_graph_mtp_steps(row, next_tokens, T, idx)` with the MTP KV index equal
to the trunk position of the source row, and `qwen4_graph_mtp_chain_step` for
the second draft. The family graph needs:

1. the `output_norm`ed trunk rows of the last forward kept for drafting;
2. all-row logits for 2- and 3-row verifies, and GDN state and conv history
   snapshots after rows 0 and 1 (the shared scan kernels already take them);
3. its own `mtp_steps` and `chain_step`, redirected at the first line of the
   Qwen3.8 functions, using the Q4_K multi-row matvec for the head;
4. the recipe tool merging `blk.64` from the MTP GGUF without its duplicated
   `token_embd` and `output`.

Flash Next fills the MTP layer's KV cache only from drafting positions;
llama.cpp also runs the MTP layer over the prompt. Which one gives better
acceptance on the 27B is measured, not assumed.

## References

- Issues and PRs: antirez/ds4 #844 (Qwen 3.8 27B + DFlash, Metal, open),
  #866 (27B request), #1075 (Bonsai 2 RFC); llama.cpp #22673, #29869, #29110,
  #30174, #31037, #30357; mlx-lm #990, #1623, #1740, #1807, #1871; mlx #3764.
- MTPLX: https://github.com/youssofal/MTPLX
