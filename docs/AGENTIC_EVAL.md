# Agentic evaluation (fork, in progress)

The continuation fixtures in `gguf-tools/quality-testing` measure how closely
a quant reproduces its reference. They do not say how well it works as an
agent, and no published benchmark covers the exact files we run. This
harness scores each quant on agent tasks, on rented CUDA machines, so models
and quants can be compared on equal terms.

Models (`agent-bench/models.tsv`, pinned by revision and sha256):

| Name | File | Engine on the pod |
| --- | --- | --- |
| `qwen38-27b-c6` | our C6 recipe (17.37 GB, private repo) | llama.cpp |
| `qwen38-27b-q4km` | ggml-org Q4_K_M (18.97 GB) | llama.cpp |
| `bonsai2-pq2` | PrismML Ternary Bonsai 2 PQ2_0 (7.21 GB) | PrismML llama.cpp |
| `flash-next-q2` | antirez Qwen3.8 Flash Next Q2 (137 GiB, 41.7 GiB resident) | ds4 CUDA |

Flash Next runs on ds4 because its padded Q2_K expert layout loads nowhere
else. The 27B family runs on llama.cpp because the fork's qwen35 code is
Metal-only.

## Suites

| Suite (`run.sh` name) | Tasks | Default runs | Measures |
| --- | ---: | ---: | --- |
| SWE-rebench leaderboard, issues created 2026-03-01..05-12 (`swe-rebench`) | 110 | 3 | Real GitHub issues, Python; likely after the training cutoff |
| Terminal-Bench 2.1 (`terminal-bench`) | 89 | 1 | Multi-step terminal work; 2.1 fixes 26-28 of 2.0's tasks |
| Terminal-Bench 4.0.0 (`terminal-bench-4`) | 63 (+3 GPU) | 1 | Released 2026-08-26, after the Qwen3.8 release; 8 h per task |
| ForecastBench, sets due 2026-08-16..10-25 (`forecastbench`) | 600 of 3,000 questions | 1 | Forecasting events after the cutoff; Brier on resolved items |
| τ²-bench retail, airline and τ³-Banking (`tau2`) | 90 of 261 | 1 (pass^1) | Tool use under policy, with a simulated customer |

Task lists are pinned in `agent-bench/tasks`. The Terminal-Bench sets are
Harbor packages that `pod_setup.sh` downloads once and checks against the
recorded tree hash (`treehash.py`); their Docker images are pinned by digest.
The pilot lists hold 20 SWE-rebench, 20 Terminal-Bench 2.1 (4 easy, 16
medium), the 20 CPU-only Terminal-Bench 4 tasks with the shortest expert time
estimate, 10 ForecastBench questions per set, and 45 τ² tasks.

τ² runs at commit 19f13ea (v2.0.0 + 112 commits; task data identical to
v2.0.0) as a custom run, not v2's released `text` benchmark (airline, retail,
telecom in six languages, user simulator gpt-5.5 at reasoning low, about $0.15
per task). Scores compare quants with each other, not with the leaderboard.
Retail's NL assertions are graded by v2's judge, gpt-5.4-mini-2026-03-17 at
reasoning low, routed through OpenRouter by `agent-bench/tau2_run.py`.

OpenRouter spend is capped at about $10 for all models. The τ² customer is
GPT-4.1 (the 1.x protocol), about $0.018 per task with prompt caching, so the
suite runs 30 tasks per domain (`tasks/tau2-budget.json`, sha256 order), one
trial: about $1.6 per model. `TAU2_FULL=1` runs all 261 tasks, 4 trials
(about $70 per model). The judges (gpt-oss-120b) cost cents. Each run ends
with its OpenRouter spend; keys load from `.env` (`OPENROUTER_KEY`,
`HF_TOKEN`, and `C6_REPO` for the private C6 repo) when not in the environment.

Epoch rates Terminal-Bench 4.0.0 Flawed: its review finds scoring defects in
30 of the 66 tasks (19 false positives, 11 false negatives). Quant
comparisons are paired on the same tasks, so the defects add noise rather
than bias, but absolute scores are not comparable with other harnesses.

ForecastBench (`agent-bench/forecastbench.py`) uses the benchmark's own
zero-shot prompts with market freeze values, today's date set to each set's
due date, and its probability parser (with a deterministic fallback instead
of an extraction model). Forecasts are stored, so a run is rescored for free
as later resolution sets are published (`fetch --resolutions-ref`, then
`score`). 2,519 items of the first five sets had resolved by 2026-11-03; the
2026-10-25 set had none yet. A question costs about 6.5K generated tokens
(Bonsai 2, mostly thinking), so the suite takes `FB_PER_SET` (default 100)
questions per set, the same sha256-ordered subset for every model: 600
questions, 501 resolved items. `FB_PER_SET=0` runs all 3,000.

### Muse Glimmer evaluation set (to add)

Meta reported Muse Glimmer 30B on the evals below (model card and the
methodology report, research.meta.ai/static/muse-glimmer-methodology). We
add them for comparability, cutoff notwithstanding; most predate Qwen3.8.

| Eval | How we run it (`run.sh` suite) | Status |
| --- | --- | --- |
| AIME 2026 (10 epochs), GPQA Diamond (4), WMDP Bio/Chem, LAB-Bench ProtocolQA, SciCode (with background), HLE text (500, judged) | Inspect AI 0.3.287 + inspect_evals 0.27.0 (`inspect`, `tasks/inspect.tsv`) | wired; GPQA and HLE need an HF token with the datasets' terms accepted; HLE is graded by inspect_evals' own two OpenRouter judges (GLM-5.3 Flash pinned to fp8, Gemini 3.7 Flash as a second column; `HLE_GRADER` overrides the first) |
| SkillsBench (87, 4 attempts) | Harbor `benchflow/skillsbench`, pinned copy (`skillsbench`) | wired |
| SWE-Bench Pro v1 (fixed 150 of 731, 4 attempts) | Harbor `scale-ai/swe-bench-pro`, pinned copy (`swe-bench-pro`; `SWEPRO_FULL=1` for all) | wired |
| τ3-Banking (97) | tau2-bench `banking_knowledge`, `bm25_grep` retrieval as AA runs it (`tau2`) | wired |
| IFBench (299 prompts, 4 runs) | own generation (`ifbench.py`), IFBench's own verifiers at a pinned commit (`ifbench`) | wired; scoring untested (needs IFBench's code, installed by pod_setup.sh) |
| AA-LCR v1.1 (100 questions, 70-115K-token prompts) | own runner (`aalcr.py`) with AA's prompt and v1.1 judge prompts; judge `JUDGE_MODEL` (`aalcr`) | wired; `run.sh` raises the engine context to 160K per session for it, runs 2 at a time |
| BEAM-128K | needs its probe and rubric pipeline | later |
| DeepSearchQA, MCP Atlas, Gaia2, WildClawBench, GDPval-AA, CI Memories, Siren AgentDojo | own harnesses; need search APIs, MCP server keys, judges or attacker models | later |
| CharXiv, ScreenSpot Pro, OmniDocBench, MMMU Pro, OSWorld-Verified | need image input | blocked until vision is served |
| VCT, MBCT, HPCT | SecureBio, access on request | blocked on access |

Our sampling stays Qwen's (top_k 20); Meta ran Glimmer with top_k 64 and the
Qwen3.6 comparison with top_k 20. Our τ² customer is GPT-4.1 for every
domain; AA's τ3-Banking numbers use GPT-5.4 Mini, so ours are comparable
between quants, not with AA's.

CWE-bench (cwe-bench.com) was considered and is not runnable: its 120 tasks
are private to Collinear and Artificial Analysis.

- SWE-rebench and Terminal-Bench run through Harbor with its terminus-2
  agent, one harness for both. τ²-bench runs its own CLI; its customer is
  one fixed API model (`openrouter/openai/gpt-4.1`, temperature 0) for every
  quant.
- Sampling is Qwen's thinking mode, as in Qwen's agent evaluations:
  temperature 1.0, top_p 0.95, top_k 20.
- Qwen3.8 publishes no training cutoff (the 27B appeared 2026-08-05). The
  110 SWE-rebench tasks are the newest the leaderboard set offers (its last
  update, 2026-07-28, ends in May). SWE-rebench V2 ends in October 2025 and
  is public training data, so it is no decontamination source.
- With paired tasks and roughly 15% of results flipping between runs, the
  smallest detectable difference is about `2*sqrt(0.15/n)`: about 7 points
  for 110 tasks (5-6 with three runs, which are correlated) and 8 for
  Terminal-Bench, and 8 for τ² at the budget size (90 tasks, one trial).
  pass^k, which catches quant damage that shows up as inconsistency, needs
  `TAU2_FULL=1` and its ~$70 per model.

## Running

On a Linux machine with CUDA, Docker and this checkout:

```sh
C6_REPO=org/private-repo HF_TOKEN=... agent-bench/pod_setup.sh qwen38-27b-c6
agent-bench/equiv.sh qwen38-27b-c6
agent-bench/run.sh swe-rebench qwen38-27b-c6 --pilot
RCLONE_DEST=r2:bench/agent-bench agent-bench/sync.sh loop &
agent-bench/run.sh tau2 qwen38-27b-c6   # keys from the environment or .env
```

`pod_setup.sh` installs missing packages, builds the engines at pinned
commits (llama.cpp `d02b4c0`, PrismML `1180918`, ds4 from this checkout),
downloads and verifies the GGUFs, and installs Harbor 0.25.0 and τ²-bench
`19f13ea` in separate venvs (τ² pins litellm below 1.82.7, Harbor needs
1.92 or later). On macOS it builds the Metal engines for local smoke tests;
`run.sh --limit N` runs only the first N tasks. `--pilot` is the smoke test:
20 IFBench prompts, 10 Inspect samples per task and 10 ForecastBench
questions per set, but one question each for AA-LCR, GPQA Diamond and
ProtocolQA, and one fixed HLE question (`hle_bf965694`, answered in about 29K
tokens so it fits the 32K cap and reaches both judges; `HLE_PILOT_SAMPLE`
picks another). On an M4 Pro the first local pilot spent 8.5 h of GPU time,
4 h 10 min of it on HLE; with these cuts a full local pilot is about 3 h.
SWE-Bench Pro and SkillsBench always hit their agent time limits on the Mac
(50 and 15 min), so locally they only prove the plumbing. `run.sh` starts the
engine and the logging proxy, runs the suite and stops them. Concurrency, context and sampling are environment variables
in `agent-bench/common.sh`. Defaults: 8 sessions, a 256K-token llama.cpp KV
pool shared by all slots (`--kv-unified`, f16 as in ds4), 128K per ds4
session.

Each run writes `RESULTS_DIR/MODEL/SUITE-TIMESTAMP/`: `manifest.json`
(file hash, engine commits, sampling, GPU), the harness output (Harbor jobs
and trajectories, τ² simulations), the engine log and the compressed request
log. A run is well under 1 GB, mostly logprobs. `sync.sh` copies results
with rclone; give it a write-only key, since the host is not ours.

## Equivalence check

A score from a CUDA engine stands for the quant only if that engine computes
the same distributions as our reference. `equiv.sh` scores the official
fixtures on the pod and compares them with `agent-bench/reference`: ds4 on
Metal for the 27B family, ds4 CUDA on a DGX Spark for Flash Next (the
`cuda-spark-20260915` tables). The gate is a relative NLL change of at most
0.5% on both sets; quants differ by 1.5-25%. The C6 and Flash Next
references are in place. Still missing, to be scored on Metal when the GPU
is free: the plain ggml-org Q4_K_M (only the Q6_K-to-Q8_0 variant was
scored) and Bonsai PQ2_0. The `qwen38-27b-ionstream-fp8-*` fixtures are not
tracked; copy them to the pod.

## Request log and replay

Every client of the model (Harbor's LiteLLM, τ², Inspect) and the proxy use
`REQUEST_TIMEOUT` (default 14400 s). The OpenAI SDK default of 600 s is far
below a long thinking turn on one GPU: the client resends the request, the
resend lands in the other session slot and evicts its prompt cache, and in
the local smoke test half the generated tokens went to resends. Inspect also
gets `--max-retries 3` instead of unlimited retries. When a client does give
up, the proxy closes its upstream connection, and ds4-server (like
llama-server) cancels the generation instead of holding the slot.

`agent-bench/proxy.py` sits between the harness and the engine. It forwards
each call, requests top-20 logprobs (and strips them again unless the client
asked), and appends one JSON line per call (status 499 when the client hung up):
the client request, the response or stream chunks, timing, and the prompt as llama-server's
`/apply-template` renders it. ds4-server returns no logprobs and has no
`/apply-template`; its prompts are rendered offline from the logged
messages.

Speed changes made later (half activations, KV precision) are judged without
rerunning the suites: teacher-force 20-30 logged trajectories through the
baseline and the candidate, and compare full-vocabulary KLD, top-1 agreement
and the NLL of the recorded actions at the model's own turns. A change whose
KLD stays far below the quant's own loss cannot move the agent scores
measurably. `agent-bench/test_proxy.py` checks the proxy without a model.

## Machines and cost

Estimates, not measurements: about 15M prefill and 7.5M generated tokens
per model for SWE-rebench-scale work, $5-15 per 27B model on a 24-48 GB card
with batching, $25-45 for Flash Next, which ds4 decodes one session at a
time on a single CUDA card. Any provider with a full VM works (Vast.ai,
Massed Compute, Lambda); on container-only providers (RunPod) the Docker
tasks need a separate VM. Run the pilot first: it measures tokens per task,
sessions per card and throughput, and checks that ds4's CUDA build runs on
the rented architecture (it has run on Ada and sm_121, not yet sm_120).
