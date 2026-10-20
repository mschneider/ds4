# Maintaining the Bonsai fork

[Bonsai 2 27B](BONSAI2_27B.md)

This repository is a long-lived community fork of
[antirez/ds4](https://github.com/antirez/ds4) that adds Ternary Bonsai 2. The
rules below keep the cost of following upstream small.

## Layout rule: fork code lives in fork-owned files

| File | Contents |
|---|---|
| `ds4_bonsai2_shape.inc` | model shape, `ds4_model_is_bonsai2()` |
| `ds4_bonsai2_model.inc` | GGUF/Hadamard validation, weight binding, layout checks, host PQ2_0 rows, engine gate |
| `ds4_bonsai2.inc` | Metal forward graph (on the Qwen3.8 graph struct and kernels) |
| `ds4_bonsai2_batch.inc` | batched multi-session decode (needs `struct ds4_session`, so it is included later than the graph) |
| `ds4_bonsai2_gpu.h`, `ds4_metal_bonsai2.m`, `metal/bonsai2.metal` | kernel API, wrappers, kernels |
| `ds4_metal_internal.h` | trampolines exposing `ds4_metal.m` statics to out-of-tree kernel files |
| `fork.mk` | build rules and fork test targets (included from `Makefile`) |
| `tests/bonsai2/` | kernel tests, golden reference, reference checker |
| `download_bonsai2.sh`, `docs/BONSAI2_27B.md`, this file | |

Upstream files get only **contact points**: one- to three-line hooks that
include or dispatch into the files above. When new Bonsai code is needed, put
it in a fork file and add a hook. Never edit shared logic inline.

## Contact points

Check this list after every upstream merge (`git diff upstream/main -- <file>`).
If a hook's surrounding function was refactored upstream, re-place the hook.
Don't fork the refactored code.

| File | Hook |
|---|---|
| `Makefile` | `include fork.mk` after the platform `CORE_OBJS` lists |
| `ds4.c` | `DS4_VARIANT_BONSAI2 = 200`. Fork variants start at 200 so they never share a KV-cache model id with upstream. |
| | `#include "ds4_bonsai2_shape.inc"` after `ds4_qwen4_layer_is_nextn` |
| | `gguf_types[142]` and `DS4_TENSOR_PQ2_0 = 142` |
| | `weights_validate_bonsai2_layout` prototype + dispatch in `weights_validate_layout` |
| | `#include "ds4_bonsai2_model.inc"` before `config_validate_model`, plus its `qwen35` arch dispatch |
| | Bonsai branches in `weights_bind_output` and `weights_bind_layer` |
| | first lines of `qwen4_graph_alloc` and `qwen4_graph_forward_tokens` redirect to Bonsai |
| | `#include "ds4_bonsai2.inc"` after the Qwen3.8 GPU section (`generate_qwen4_metal_argmax`) |
| | `ds4_engine_routed_quant_bits`: Bonsai reports 2. Without it the KV disk cache never stores Bonsai payloads. |
| | `qwen4_ref_row`: `DS4_TENSOR_PQ2_0` case |
| | `ds4_engine_open_internal`: `bonsai2_engine_supported` gate |
| | `#include "ds4_bonsai2_batch.inc"` and a first-line redirect in `qwen4_graph_encode_native_session_batch` |
| `ds4.h` | `ds4_engine_is_bonsai2()` |
| `ds4_metal.m` | include of `ds4_metal_internal.h`, fork sources appended in `ds4_gpu_full_source`, trampoline block at the end |
| `ds4_server.c` | model id `ternary-bonsai-2-27b` in `server_model_id_from_engine`, `server_model_alias_known`, `send_models` |
| `ds4_cli.c` | `<|im_start|>` in `is_rendered_chat_prompt`. This is a generic Qwen fix and could go upstream. |

## Coupling to the Qwen3.8 code

Bonsai reuses these Qwen3.8 internals. An upstream change to any of them can
break Bonsai silently: it still compiles, but produces wrong tokens.

- `ds4_qwen4_gpu_graph` fields and `DS4_QWEN4_SCRATCH_FIELDS`, `qwen4_graph_free`,
  `qwen4_graph_alloc_f32`, and the session/payload code that walks the graph.
- Kernels: `ds4_gpu_qwen4_conv_stream_tensor`, `_gdn_prep_tensor`,
  `_gdn_scan_tensor` (tiled `h % Hk` V order, `ssm_a = -exp(A_log)`), and
  `_attn_decode_tensor` in dense mode. `kernel_bonsai2_attn_prep` calls
  `qwen4_attn_prep_slot` from `metal/qwen4.metal`. Its argument struct must
  match `ds4_metal_args_qwen4_attn_prep`.
- `g_qwen4_rope_freq` / `g_qwen4_rope_mscale` / `qwen4_rope_configure`, plus
  `qwen4_gemv` (prefill alpha/beta) and `qwen4_graph_apply_steering_*`.
- Vision and batching: `qwen4_mrope_pos`, `qwen4_span_row`,
  `qwen4_fake_spans`, the `vis_spans` and `mrope_delta` graph fields,
  `qwen4_row_view`, `qwen4_batch_scratch_ensure`, `batch_logits`, the
  checks in `qwen4_graph_native_session_batch_check`, and the batch driver
  that reads `batch_logits` and advances `pos`.

`make test-bonsai2-reference` catches all of these, so run it after every merge.

## Upstream sync procedure

```sh
git fetch upstream
git switch bonsai2
git merge upstream/main          # merge, never rebase: rerere and history stay usable
make && make test-bonsai2-kernels && make test-bonsai2-reference
make ds4_test ds4_agent_test && ./ds4_test --server && ./ds4_agent_test
git push origin bonsai2
```

`rerere.enabled` is set in this clone. Recorded conflict resolutions in
contact points are replayed on later merges. Set it in new clones with
`git config rerere.enabled true`.

## Release checks

1. `make test-bonsai2-kernels`, which needs no model.
2. `make test-bonsai2-reference`, which must report "all cases match".
3. Speed, back to back with the reference fork (`misc/bonsai2/` holds the
   local reference build, which is not committed):
   `./ds4-bench -m <gguf> --prompt-file <text> --ctx-start 512 --ctx-max 4096 --step-mul 2`.
4. A server smoke test covering `/v1/models`, a chat with reasoning, a tool
   call, a follow-up turn (prefix cache hit), and a restart with
   `--kv-disk-dir` (disk cache hit).

## Regenerating the reference

The reference comes from PrismML's llama.cpp fork (`prism` branch). If
PrismML publishes new weights or changes the format:

1. Pin the new HF revision and sha256 in `download_bonsai2.sh` and in
   `golden.json`.
2. Rebuild the fork and re-run its server greedily (temp 0, top-k 1,
   `n_probs` 20, 128 tokens) on the cases in `golden.json`.
3. Rewrite `golden.json`, keeping the same compact format.
