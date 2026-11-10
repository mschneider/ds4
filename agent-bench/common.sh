# Shared settings for the agent-bench scripts (sourced, not run).
# Override any of these in the environment.

BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$BENCH_DIR/.." && pwd)
WORK=${WORK:-/workspace/agent-bench}        # engines, venvs, models, results
MODELS_DIR=${MODELS_DIR:-$WORK/models}

# API keys come from the environment, or else from the checkout's .env
# (OPENROUTER_KEY=..., HF_TOKEN=...; .env is excluded from git).
env_key() { [ -f "$REPO_DIR/.env" ] && sed -n "s/^$1=//p" "$REPO_DIR/.env" | tail -1 | tr -d '"'"'"'\r'; }
[ -n "${OPENROUTER_API_KEY:-}" ] || OPENROUTER_API_KEY=$(env_key OPENROUTER_KEY)
[ -n "${HF_TOKEN:-}" ] || HF_TOKEN=$(env_key HF_TOKEN)
[ -n "${C6_REPO:-}" ] || C6_REPO=$(env_key C6_REPO)     # private repo holding the C6 GGUF
[ -n "$OPENROUTER_API_KEY" ] && export OPENROUTER_API_KEY || unset OPENROUTER_API_KEY
[ -n "$HF_TOKEN" ] && export HF_TOKEN || unset HF_TOKEN
[ -n "$C6_REPO" ] && export C6_REPO || unset C6_REPO
RESULTS_DIR=${RESULTS_DIR:-$WORK/results}

# Engines, pinned.  ds4 builds from this checkout.
LLAMA_CPP_URL=https://github.com/ggml-org/llama.cpp
LLAMA_CPP_REF=${LLAMA_CPP_REF:-d02b4c0d66796e0a44380611d038123fefb9fe83}
PRISM_URL=https://github.com/PrismML-Eng/llama.cpp
PRISM_REF=${PRISM_REF:-11809184898113482c2f510fd13da669aaba1052}
TAU2_URL=https://github.com/sierra-research/tau2-bench
TAU2_REF=${TAU2_REF:-19f13ea3f5fba8b4c4d272d1353590f29dc53062}
HARBOR_VERSION=${HARBOR_VERSION:-0.25.0}
INSPECT_AI_VERSION=${INSPECT_AI_VERSION:-0.3.287}
INSPECT_EVALS_VERSION=${INSPECT_EVALS_VERSION:-0.27.0}
# LLM judge for AA-LCR, the same for every quant (HLE uses its own judges).
JUDGE_MODEL=${JUDGE_MODEL:-openrouter/openai/gpt-oss-120b}

# Serving: the engine on SERVER_PORT, the logging proxy on PROXY_PORT.
SERVER_PORT=${SERVER_PORT:-8080}
PROXY_PORT=${PROXY_PORT:-9000}
# A 24 GB card cannot hold the 256K pool next to the weights (the first Vast
# pilot ran 128K for 8 slots and eight agent sessions exhausted it: every
# "Context size has been exceeded" was counted as an infrastructure error).
# Below 40 GB the defaults become 4 sessions on a 192K pool (20.8 GB measured
# on an RTX 4090 with Bonsai 2); explicit CTX/PARALLEL still win.
if [ -z "${CTX:-}${PARALLEL:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
    if [ -n "$vram_mib" ] && [ "$vram_mib" -lt 40000 ]; then
        CTX=196608 PARALLEL=4
    fi
fi
CTX=${CTX:-262144}          # llama.cpp: one KV pool shared by all slots
PARALLEL=${PARALLEL:-8}     # concurrent sessions served
DS4_CTX=${DS4_CTX:-131072}  # ds4: per session
TOP_LOGPROBS=${TOP_LOGPROBS:-20}

# Qwen3.8 thinking-mode sampling, as in Qwen's own agent evaluations.
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-131072}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-32768}
# Harbor's agent timeouts are wall-clock and sized for datacenter APIs. One
# GPU serving 4-8 thinking sessions decodes ~60-70 tok/s per session, so at
# 1.0 most Terminal-Bench trials timed out after 3-16 steps (Vast pilot,
# 2026-11-08); 4.0 gives the same number of turns a fast API would get.
AGENT_TIMEOUT_MULT=${AGENT_TIMEOUT_MULT:-4.0}
# Per-request timeout for every client of the model (and the proxy). The
# OpenAI SDK default of 600 s makes LiteLLM/Inspect resend long thinking
# requests while the abandoned one keeps generating in its slot.
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-14400}

die() { echo "agent-bench: $*" >&2; exit 1; }

# model_field NAME COLUMN: one field of agent-bench/models.tsv
# (1 name, 2 engine, 3 repo, 4 revision, 5 file, 6 sha256).
model_field() {
    local v
    v=$(awk -F'\t' -v n="$1" -v c="$2" '$1 == n { print $c }' "$BENCH_DIR/models.tsv")
    [ -n "$v" ] || die "unknown model '$1' (see agent-bench/models.tsv)"
    v=${v//'${C6_REPO}'/${C6_REPO:-}}
    echo "$v"
}

model_path() { echo "$MODELS_DIR/$(model_field "$1" 5)"; }

# model_engine NAME: the engine from models.tsv, or ENGINE to override it
# (e.g. ENGINE=ds4 with DS4_SERVER from a bonsai2 build for local Metal runs).
model_engine() { echo "${ENGINE:-$(model_field "$1" 2)}"; }
DS4_SERVER=${DS4_SERVER:-$REPO_DIR/ds4-server}
