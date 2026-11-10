#!/usr/bin/env bash
# Prepare a rented Linux CUDA machine for the agent benchmarks:
#   agent-bench/pod_setup.sh MODEL...
# Installs missing system packages (as root), builds the engines the given
# models need, downloads and verifies the GGUFs, and installs Harbor and
# tau2-bench.  Safe to rerun; finished steps are skipped.  On macOS it builds
# the Metal engines instead, for local test runs; a GGUF already placed in
# MODELS_DIR (or linked there) is verified instead of downloaded.
set -euo pipefail
source "$(dirname "$0")/common.sh"

[ $# -gt 0 ] || die "usage: pod_setup.sh MODEL... (names in agent-bench/models.tsv)"
for m in "$@"; do model_field "$m" 2 >/dev/null; done
mkdir -p "$WORK" "$MODELS_DIR" "$RESULTS_DIR"

need_pkgs=()
os=$(uname -s)
pkgs="git cmake make g++ curl python3 zstd docker"
[ "$os" = Linux ] && pkgs+=" rclone"     # sync.sh; local macOS runs don't sync
for c in $pkgs; do
    command -v "$c" >/dev/null || need_pkgs+=("$c")
done
if [ ${#need_pkgs[@]} -gt 0 ]; then
    [ "$os" = Linux ] || die "missing: ${need_pkgs[*]}"
    [ "$(id -u)" = 0 ] || die "missing: ${need_pkgs[*]} (install them or rerun as root)"
    apt-get update -qq
    for c in "${need_pkgs[@]}"; do
        case $c in
            g++) apt-get install -y -qq build-essential ;;
            docker) apt-get install -y -qq docker.io ;;
            python3) apt-get install -y -qq python3 python3-venv python3-pip ;;
            *) apt-get install -y -qq "$c" ;;
        esac
    done
fi
if ! docker info >/dev/null 2>&1; then
    # The Harbor suites need the daemon; local macOS runs of the single-turn
    # suites and tau2 do not, and Docker Desktop's VM would take unified memory
    # from the model.
    [ "$os" = Darwin ] && echo "warning: Docker daemon not reachable; Harbor suites will not run" ||
        die "Docker daemon not reachable; the agent tasks need it"
fi
if [ "$os" = Linux ] && [ "$(id -u)" = 0 ] && [ -f /etc/os-release ]; then
    . /etc/os-release
    # Ubuntu's unattended-upgrades replaced the NVIDIA user-space libraries
    # under a running engine on the first pod (driver/library mismatch, every
    # suite failed). Freeze the driver packages and stop the timer.
    systemctl disable --now unattended-upgrades >/dev/null 2>&1 || true
    held=$(dpkg -l 2>/dev/null | awk '/^ii .*nvidia/ { print $2 }')
    [ -n "$held" ] && apt-mark hold $held >/dev/null 2>&1 || true
    # Rented VM images often ship the driver but no toolkit: add NVIDIA's apt
    # repo and install CUDA 12.8 (works with any driver from 570 on).
    if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ] && [ "${ID:-}" = ubuntu ]; then
        tag=ubuntu$(echo "$VERSION_ID" | tr -d .)
        curl -fsSL -o /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/$tag/x86_64/cuda-keyring_1.1-1_all.deb"
        dpkg -i /tmp/cuda-keyring.deb >/dev/null && apt-get update -qq && apt-get install -y -qq cuda-toolkit-12-8
    fi
    # tau2-bench needs Python 3.12; Ubuntu 22.04 ships 3.10.
    if ! command -v python3.12 >/dev/null && ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null && [ "${ID:-}" = ubuntu ]; then
        apt-get install -y -qq software-properties-common >/dev/null
        add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1 && apt-get update -qq
        apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
    fi
fi
if [ "$os" = Darwin ]; then
    gpu_flags=() ds4_target=""          # Metal is the default on macOS
else
    command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ] || die "CUDA toolkit (nvcc) not found"
    export PATH=/usr/local/cuda/bin:$PATH
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    gpu_flags=(-DGGML_CUDA=ON)
    # The Makefile's Blackwell builds (compute_120a/121a with MXFP4) need the
    # named arch; `cuda-generic` compiles for `native` and would skip them.
    case $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1) in
        12.0) ds4_target="cuda CUDA_ARCH=sm_120" ;;
        12.1) ds4_target=cuda-spark ;;
        *) ds4_target=cuda-generic ;;
    esac
fi
jobs=$(getconf _NPROCESSORS_ONLN)
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# build_llama DIR URL REF: CUDA (Metal on macOS) build of a llama.cpp tree at a pinned commit.
build_llama() {
    local dir=$1 url=$2 ref=$3
    if [ ! -x "$dir/build/bin/llama-server" ] || [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" != "$ref" ]; then
        [ -d "$dir/.git" ] || git clone -q --filter=blob:none "$url" "$dir"
        git -C "$dir" fetch -q origin "$ref" && git -C "$dir" checkout -q "$ref"
        cmake -S "$dir" -B "$dir/build" ${gpu_flags[@]+"${gpu_flags[@]}"} -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
        cmake --build "$dir/build" -j"$jobs" --target llama-server llama-cli llama-perplexity >/dev/null
    fi
    echo "built: $dir at $(git -C "$dir" rev-parse --short HEAD)"
}

# fetch MODEL: download with resume, then verify the sha256.
fetch() {
    local m=$1 repo rev file sha out
    repo=$(model_field "$m" 3); rev=$(model_field "$m" 4)
    file=$(model_field "$m" 5); sha=$(model_field "$m" 6)
    case $sha in @*) die "$m: fill in its sha256 in agent-bench/models.tsv" ;; esac
    out="$MODELS_DIR/$file"
    if [ -f "$out.ok" ]; then echo "model: $file verified earlier"; return; fi
    if [ ! -e "$out" ] || [ "$(sha256 "$out")" != "$sha" ]; then
        [ -n "$repo" ] || die "$m: set C6_REPO to the private Hugging Face repo"
        local auth=()
        [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
        echo "model: downloading $repo@$rev/$file"
        curl -fL --retry 5 -C - ${auth[@]+"${auth[@]}"} -o "$out" "https://huggingface.co/$repo/resolve/$rev/$file"
        [ "$(sha256 "$out")" = "$sha" ] || die "$file: sha256 mismatch"
    fi
    echo "model: $file sha256 ok"
    touch "$out.ok"
}

for m in "$@"; do
    case $(model_engine "$m") in
        llama) build_llama "$WORK/llama.cpp" "$LLAMA_CPP_URL" "$LLAMA_CPP_REF" ;;
        prism) build_llama "$WORK/prism-llama.cpp" "$PRISM_URL" "$PRISM_REF" ;;
        ds4) if [ "$DS4_SERVER" = "$REPO_DIR/ds4-server" ]; then
                 make -C "$REPO_DIR" -s $ds4_target && echo "built: ds4 at $(git -C "$REPO_DIR" rev-parse --short HEAD)"
             else
                 [ -x "$DS4_SERVER" ] || die "DS4_SERVER=$DS4_SERVER is not executable"
                 echo "using: $DS4_SERVER"
             fi ;;
    esac
    fetch "$m"
done

# Harness: Harbor (SWE-rebench, Terminal-Bench) and tau2-bench, in separate
# venvs: tau2 pins an older litellm than Harbor accepts.
PY=$(command -v python3.12 || command -v python3.13 || command -v python3)
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 12))' || die "tau2-bench needs Python >= 3.12"
if [ ! -x "$WORK/venv/bin/harbor" ]; then
    "$PY" -m venv "$WORK/venv"
    "$WORK/venv/bin/pip" install -q "harbor==$HARBOR_VERSION"
fi
if [ ! -d "$WORK/tau2-bench/.git" ]; then
    git clone -q "$TAU2_URL" "$WORK/tau2-bench"
fi
git -C "$WORK/tau2-bench" checkout -q "$TAU2_REF"
[ -x "$WORK/tau2-venv/bin/python" ] || "$PY" -m venv "$WORK/tau2-venv"
"$WORK/tau2-venv/bin/pip" install -q -e "$WORK/tau2-bench[knowledge]"   # + banking_knowledge retrieval
# tau2 imports audioop, which Python 3.13 removed.
"$WORK/tau2-venv/bin/python" -c 'import sys; sys.exit(sys.version_info < (3, 13))' &&
    "$WORK/tau2-venv/bin/pip" install -q audioop-lts
# Inspect AI for the single-turn evals (agent-bench/tasks/inspect.tsv).
if [ ! -x "$WORK/inspect-venv/bin/inspect" ]; then
    "$PY" -m venv "$WORK/inspect-venv"
    "$WORK/inspect-venv/bin/pip" install -q "inspect-ai==$INSPECT_AI_VERSION" "inspect-evals==$INSPECT_EVALS_VERSION" openai
fi
# inspect-ai 0.3.287 imports acp.helpers, dropped in agent-client-protocol 1.0.0.
"$WORK/inspect-venv/bin/python" -c 'import acp.helpers' 2>/dev/null ||
    "$WORK/inspect-venv/bin/pip" install -q "agent-client-protocol<1"
# SciCode fetches its test data from Google Drive.
"$WORK/inspect-venv/bin/python" -c 'import gdown' 2>/dev/null ||
    "$WORK/inspect-venv/bin/pip" install -q "gdown==5.2.0"

# Task sets pinned by content: Harbor packages are downloaded once and checked
# against the tree hash recorded in agent-bench/tasks; run.sh uses these copies.
fetch_harbor() {   # fetch_harbor NAME PACKAGE SUBDIR
    local dir="$WORK/datasets/$1" want got
    want=$(cat "$BENCH_DIR/tasks/$1.treehash")
    for try in 1 2 3; do   # the registry drops connections now and then
        [ -d "$dir/$3" ] && [ "$(python3 "$BENCH_DIR/treehash.py" "$dir/$3")" = "$want" ] && break
        "$WORK/venv/bin/harbor" download "$2" -o "$dir" >"$WORK/harbor-download.log" 2>&1 ||
            { echo "tasks: $1 download failed (try $try)"; sleep 10; }
    done
    got=$(python3 "$BENCH_DIR/treehash.py" "$dir/$3")
    [ "$got" = "$want" ] || die "$1: task tree hash $got != $want (package changed upstream?)"
    echo "tasks: $1 verified"
}
fetch_harbor terminal-bench-2.1 terminal-bench/terminal-bench-2-1 terminal-bench-2-1
fetch_harbor terminal-bench-4 terminal-bench/terminal-bench@4.0.0 terminal-bench
fetch_harbor skillsbench benchflow/skillsbench skillsbench
fetch_harbor swe-bench-pro scale-ai/swe-bench-pro swe-bench-pro
python3 "$BENCH_DIR/forecastbench.py" fetch "$WORK/datasets/forecastbench"
python3 "$BENCH_DIR/aalcr.py" fetch "$WORK/datasets/aalcr"
python3 "$BENCH_DIR/ifbench.py" fetch "$WORK/datasets/ifbench"
# IFBench is scored by its own verifier code, pinned to the same commit as the data.
IFBENCH_REF=cd189a9c2889526d8af901d2103fd3e1ffcea6df
if [ ! -d "$WORK/ifbench-src/.git" ]; then
    git clone -q https://github.com/allenai/IFBench "$WORK/ifbench-src"
fi
git -C "$WORK/ifbench-src" checkout -q "$IFBENCH_REF"
if [ ! -x "$WORK/ifbench-venv/bin/python" ]; then
    "$PY" -m venv "$WORK/ifbench-venv"
    "$WORK/ifbench-venv/bin/pip" install -q -e "$WORK/ifbench-src"
fi
echo "harness: harbor $("$WORK/venv/bin/harbor" --version), tau2 at ${TAU2_REF:0:7}"
echo "ready: agent-bench/serve.sh start MODEL"
