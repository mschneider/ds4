#!/usr/bin/env bash
# Run one benchmark suite against one model:
#   agent-bench/run.sh SUITE MODEL [--pilot] [--attempts K] [--concurrency N] [--limit N]
# --limit runs only the first N tasks of the list (per domain for tau2), for
# smoke tests.  SUITE: swe-rebench (the 110 issues created 2026-03..05),
# terminal-bench (Terminal-Bench 2.1, 89 tasks), terminal-bench-4 (4.0.0, 66
# tasks; the 3 GPU tasks only with TB4_GPU=1), skillsbench (87 tasks with skills
# mounted), swe-bench-pro (a fixed 150-task subset of the public 731; all with
# SWEPRO_FULL=1), forecastbench (the question sets
# due after the cutoff), ifbench (299 prompts x 4), aalcr (AA-LCR v1.1, 100
# questions of 70-115K tokens; raises the context as needed, runs 2 at a time),
# inspect (the single-turn evals in tasks/inspect.tsv;
# INSPECT_TASKS selects some) or tau2 (retail, airline and tau3 banking;
# TAU2_DOMAINS selects, TAU2_TASK_IDS picks tasks).  Results land
# in RESULTS_DIR/MODEL/SUITE-TIMESTAMP with a manifest, the harness output and
# the proxy's request log; sync.sh copies them off the machine.
set -euo pipefail
source "$(dirname "$0")/common.sh"

[ $# -ge 2 ] || die "usage: run.sh SUITE MODEL [--pilot] [--attempts K] [--concurrency N] [--limit N]"
suite=$1 model=$2
shift 2
pilot=0 attempts= conc=$PARALLEL limit=
while [ $# -gt 0 ]; do
    case $1 in
        --pilot) pilot=1 ;;
        --attempts) attempts=$2; shift ;;
        --concurrency) conc=$2; shift ;;
        --limit) limit=$2; shift ;;
        *) die "unknown option $1" ;;
    esac
    shift
done
case $suite in
    swe-rebench) attempts=${attempts:-3} ;;
    terminal-bench|terminal-bench-4|forecastbench|inspect|aalcr) attempts=${attempts:-1} ;;
    ifbench) attempts=${attempts:-4} ;;                        # Muse Glimmer: 4 runs
    skillsbench|swe-bench-pro) attempts=${attempts:-4} ;;      # Muse Glimmer: 4 attempts
    tau2) [ -n "${TAU2_FULL:-}" ] && attempts=${attempts:-4}; attempts=${attempts:-1} ;;
    *) die "unknown suite $suite (swe-rebench, terminal-bench, terminal-bench-4, skillsbench, swe-bench-pro, forecastbench, inspect, ifbench, aalcr, tau2)" ;;
esac
[ $pilot = 1 ] && attempts=1
model_field "$model" 1 >/dev/null
# AA-LCR prompts run to 115K tokens plus a 32K answer, AALCR_CONC at a time
# (default 2; 1 on a 24 GB card): make sure the engine's context covers that
# (serve.sh reads these).
if [ $suite = aalcr ]; then
    aalcr_conc=${AALCR_CONC:-2}
    [ "$conc" -gt "$aalcr_conc" ] && conc=$aalcr_conc
    [ "$DS4_CTX" -ge 163840 ] || export DS4_CTX=163840
    [ "$CTX" -ge $((163840 * conc)) ] || export CTX=$((163840 * conc))
fi

run="$RESULTS_DIR/$model/$suite-$(date -u +%Y%m%dT%H%M%SZ)"
[ $pilot = 1 ] && run+="-pilot"
mkdir -p "$run"
V="$WORK/venv/bin"
api="http://127.0.0.1:$PROXY_PORT/v1"
export OPENAI_API_KEY=${OPENAI_API_KEY:-sk-local}   # litellm requires one; the server ignores it

python3 - "$run/manifest.json" <<EOF
import json, subprocess, sys, time
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception: return None
json.dump({
    "suite": "$suite", "model": "$model", "pilot": bool($pilot), "limit": ${limit:-None}, "attempts": $attempts,
    "concurrency": $conc, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "engine": "$(model_engine "$model")", "ds4_server": "$DS4_SERVER",
    "ds4_server_commit": sh("git -C '$(dirname "$DS4_SERVER")' rev-parse HEAD"), "hf_repo": "$(model_field "$model" 3)",
    "revision": "$(model_field "$model" 4)", "file": "$(model_field "$model" 5)",
    "sha256": "$(model_field "$model" 6)",
    "ds4_commit": sh("git -C '$REPO_DIR' rev-parse HEAD"),
    "llama_cpp_ref": "$LLAMA_CPP_REF", "prism_ref": "$PRISM_REF",
    "tau2_ref": "$TAU2_REF", "harbor": sh("'$V/harbor' --version"),
    "sampling": {"temperature": $TEMPERATURE, "top_p": $TOP_P, "top_k": $TOP_K},
    "ctx": $CTX, "ds4_ctx": $DS4_CTX, "max_output_tokens": $MAX_OUTPUT_TOKENS, "agent_timeout_mult": $AGENT_TIMEOUT_MULT, "parallel": $PARALLEL,
    "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
    "host": sh("hostname"),
}, open(sys.argv[1], "w"), indent=1)
EOF

# OpenRouter spend (simulated customer, judges) is reported per run.
or_usage() {
    [ -n "${OPENROUTER_API_KEY:-}" ] || return 0
    curl -s -m 20 -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/key |
        python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["usage"])' 2>/dev/null || true
}
or_start=$(or_usage)

"$BENCH_DIR/serve.sh" start "$model" "$run"
trap '"$BENCH_DIR/serve.sh" stop' EXIT

model_info=$(printf '{"max_input_tokens":%d,"max_output_tokens":%d,"input_cost_per_token":0,"output_cost_per_token":0}' \
    "$MAX_INPUT_TOKENS" "$MAX_OUTPUT_TOKENS")
call_kwargs=$(printf '{"top_p":%s,"extra_body":{"top_k":%d},"timeout":%d}' "$TOP_P" "$TOP_K" "$REQUEST_TIMEOUT")

harbor_run() {   # harbor_run DATASET|DIR TASKLIST [NAME_PREFIX] [EXCLUDE_LIST]
    local include=() t n=0 prefix=${3:-} src=(-d "$1")
    [ -d "$1" ] && src=(-p "$1")    # a pinned local copy from pod_setup.sh
    while read -r t; do
        [ -n "$t" ] || continue
        [ -n "${4:-}" ] && grep -qx "$t" "$4" && continue
        [ -n "$limit" ] && [ $n -ge "$limit" ] && break
        include+=(-i "$prefix$t"); n=$((n + 1))
    done <"$2"
    "$V/harbor" run "${src[@]}" -a terminus-2 -m "openai/$model" \
        --ak "api_base=$api" --ak "temperature=$TEMPERATURE" \
        --ak "llm_call_kwargs=$call_kwargs" --ak "model_info=$model_info" \
        --agent-timeout-multiplier "$AGENT_TIMEOUT_MULT" \
        "${include[@]}" -n "$conc" -k "$attempts" -o "$run/jobs" --job-name "$suite" -y -q \
        2>&1 | tee "$run/harness.log"
}

case $suite in
    swe-rebench)
        list=$BENCH_DIR/tasks/swe-rebench-2026-03.txt
        [ $pilot = 1 ] && list=$BENCH_DIR/tasks/swe-rebench-pilot.txt
        harbor_run swe-rebench/swe-rebench-leaderboard "$list" swe-rebench/ ;;
    terminal-bench)
        list=$BENCH_DIR/tasks/terminal-bench-2.1.txt
        [ $pilot = 1 ] && list=$BENCH_DIR/tasks/terminal-bench-2.1-pilot.txt
        harbor_run "$WORK/datasets/terminal-bench-2.1/terminal-bench-2-1" "$list" ;;
    terminal-bench-4)
        list=$BENCH_DIR/tasks/terminal-bench-4.txt
        [ $pilot = 1 ] && list=$BENCH_DIR/tasks/terminal-bench-4-pilot.txt
        skip=$BENCH_DIR/tasks/terminal-bench-4-gpu.txt
        [ "${TB4_GPU:-0}" = 1 ] && skip=
        harbor_run "$WORK/datasets/terminal-bench-4/terminal-bench" "$list" "" "$skip" ;;
    skillsbench)
        list=$BENCH_DIR/tasks/skillsbench.txt
        [ $pilot = 1 ] && list=$BENCH_DIR/tasks/skillsbench-pilot.txt
        harbor_run "$WORK/datasets/skillsbench/skillsbench" "$list" ;;
    swe-bench-pro)
        list=$BENCH_DIR/tasks/swe-bench-pro-150.txt
        [ "${SWEPRO_FULL:-0}" = 1 ] && { list=$run/swe-bench-pro-all.txt; ls "$WORK/datasets/swe-bench-pro/swe-bench-pro" >"$list"; }
        [ $pilot = 1 ] && list=$BENCH_DIR/tasks/swe-bench-pro-pilot.txt
        harbor_run "$WORK/datasets/swe-bench-pro/swe-bench-pro" "$list" ;;
    ifbench)
        n=${limit:-0}
        [ $pilot = 1 ] && [ -z "$limit" ] && n=20
        python3 "$BENCH_DIR/ifbench.py" run "$WORK/datasets/ifbench" "$run/ifbench" --api "$api" --model "$model" \
            --epochs "$attempts" --limit "$n" --concurrency "$conc" --max-tokens "$MAX_OUTPUT_TOKENS" \
            --timeout "$REQUEST_TIMEOUT" --sampling "$(printf '{"temperature":%s,"top_p":%s,"top_k":%d}' "$TEMPERATURE" "$TOP_P" "$TOP_K")" \
            2>&1 | tee "$run/harness.log"
        IFBENCH_VENV=$WORK/ifbench-venv IFBENCH_SRC=$WORK/ifbench-src \
            python3 "$BENCH_DIR/ifbench.py" score "$WORK/datasets/ifbench" "$run/ifbench" 2>&1 | tee "$run/score.txt" ;;
    aalcr)
        n=${limit:-0}
        # The first question is mostly prefill with a short answer (~28 min on
        # an M4 Pro); more only repeat that, so the pilot runs one.
        [ $pilot = 1 ] && [ -z "$limit" ] && n=1
        python3 "$BENCH_DIR/aalcr.py" run "$WORK/datasets/aalcr" "$run/aalcr" --api "$api" --model "$model" \
            --epochs "$attempts" --limit "$n" --concurrency "$conc" \
            --timeout "$REQUEST_TIMEOUT" --sampling "$(printf '{"temperature":%s,"top_p":%s,"top_k":%d}' "$TEMPERATURE" "$TOP_P" "$TOP_K")" \
            2>&1 | tee "$run/harness.log"
        if [ -n "${OPENROUTER_API_KEY:-}" ]; then
            python3 "$BENCH_DIR/aalcr.py" judge "$WORK/datasets/aalcr" "$run/aalcr" --judge "$JUDGE_MODEL" 2>&1 | tee -a "$run/harness.log"
            python3 "$BENCH_DIR/aalcr.py" score "$run/aalcr" 2>&1 | tee "$run/score.txt"
        else
            echo "aalcr: no OPENROUTER_API_KEY; judge later with aalcr.py judge" | tee "$run/score.txt"
        fi ;;
    forecastbench)
        # One zero-shot forecast per question; scored offline against the
        # resolution sets (rescore later with forecastbench.py score).  A full
        # set is ~500 questions at ~6.5K thinking tokens each, so by default we
        # take FB_PER_SET questions per set (same hash-ordered subset for every
        # model); FB_PER_SET=0 runs everything.
        n=${limit:-${FB_PER_SET:-100}}
        [ $pilot = 1 ] && [ -z "$limit" ] && n=10
        # Score whatever finished, even if generation was interrupted.
        python3 "$BENCH_DIR/forecastbench.py" run "$WORK/datasets/forecastbench" "$run/forecastbench" \
            --api "$api" --model "$model" --concurrency "$conc" --limit "$n" --max-tokens "$MAX_OUTPUT_TOKENS" \
            --timeout "$REQUEST_TIMEOUT" --sampling "$(printf '{"temperature":%s,"top_p":%s,"top_k":%d}' "$TEMPERATURE" "$TOP_P" "$TOP_K")" \
            2>&1 | tee "$run/harness.log" || echo "forecastbench: generation incomplete" | tee -a "$run/harness.log"
        python3 "$BENCH_DIR/forecastbench.py" score "$WORK/datasets/forecastbench" "$run/forecastbench" \
            2>&1 | tee "$run/score.txt" ;;
    inspect)
        # Single-turn evals through Inspect AI, against the proxy.  Epochs and
        # limits come from tasks/inspect.tsv; --pilot runs 10 samples, 1 epoch
        # (one for the slow tasks: GPQA, ProtocolQA and the judged HLE question).
        export LOCAL_BASE_URL=$api LOCAL_API_KEY=$OPENAI_API_KEY
        while IFS=$'\t' read -r task epochs lim targs needs; do
            case $task in ''|'#'*) continue ;; esac
            [ -n "${INSPECT_TASKS:-}" ] && ! echo " $INSPECT_TASKS " | grep -q " $task " && continue
            [ $pilot = 1 ] && { epochs=1; lim=10; }
            case $task in gpqa_diamond|lab_bench_protocolqa) [ $pilot = 1 ] && lim=1 ;; esac
            [ -n "$limit" ] && lim=$limit
            extra=()
            if [ $pilot = 1 ] && [ $task = hle ]; then
                # One question Bonsai 2 answers in ~29K tokens (24 min on an M4
                # Pro), so the pilot reaches the judges; most need 64K or more.
                extra+=(--sample-id "${HLE_PILOT_SAMPLE:-hle_bf965694}")
            elif [ "$lim" != 0 ]; then
                extra+=(--limit "$lim")
            fi
            if [ "$targs" != - ]; then
                for kv in $(echo "$targs" | tr ',' ' '); do extra+=(-T "$kv"); done
            fi
            case $needs in *hf-gated*) [ -n "${HF_TOKEN:-}" ] || {
                echo "inspect: $task skipped: gated dataset, set HF_TOKEN (terms accepted)" | tee -a "$run/harness.log"
                continue; } ;; esac
            case $needs in *judge*) [ -n "${OPENROUTER_API_KEY:-}" ] || {
                echo "inspect: $task skipped: judged, set OPENROUTER_API_KEY" | tee -a "$run/harness.log"
                continue; }
                # HLE's own judges (inspect_evals' default run config: GLM-5.3
                # Flash pinned to fp8, Gemini 3.7 Flash as a second column), the
                # same for every quant; HLE_GRADER overrides the primary one.
                [ -n "${HLE_GRADER:-}" ] && extra+=(--model-role "grader=$HLE_GRADER") ;; esac
            echo "inspect: $task, $epochs epochs, limit $lim" | tee -a "$run/harness.log"
            # Inspect retries forever by default; see REQUEST_TIMEOUT in common.sh.
            "$WORK/inspect-venv/bin/inspect" eval "inspect_evals/$task" --model "openai-api/local/$model" \
                --epochs "$epochs" --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" \
                --max-tokens "$MAX_OUTPUT_TOKENS" --max-connections "$conc" \
                -M client_timeout=$REQUEST_TIMEOUT --max-retries 3 \
                --log-dir "$run/inspect" --log-format eval --no-fail-on-error ${extra[@]+"${extra[@]}"} \
                2>&1 | tee -a "$run/harness.log" || echo "inspect: $task failed" | tee -a "$run/harness.log"
        done <"$BENCH_DIR/tasks/inspect.tsv" ;;
    tau2)
        # The simulated customer is one fixed API model for every quant.
        [ -n "${OPENROUTER_API_KEY:-}" ] || die "tau2 needs OPENROUTER_API_KEY for the user simulator"
        user_llm=${TAU2_USER_LLM:-openrouter/openai/gpt-4.1}
        agent_args=$(printf '{"api_base":"%s","temperature":%s,"top_p":%s,"extra_body":{"top_k":%d},"timeout":%d}' \
            "$api" "$TEMPERATURE" "$TOP_P" "$TOP_K" "$REQUEST_TIMEOUT")
        # banking_knowledge is tau3-Banking (97 tasks, 698 documents), with the
        # offline BM25 + grep retrieval Artificial Analysis uses.
        for domain in ${TAU2_DOMAINS:-retail airline banking_knowledge}; do
            retrieval=()
            [ "$domain" = banking_knowledge ] && retrieval=(--retrieval-config "${TAU2_RETRIEVAL:-bm25_grep}")
            ids=()
            # The simulated customer is paid per token (~$0.018 per task with
            # GPT-4.1), so by default each domain runs a fixed 30-task subset
            # (tasks/tau2-budget.json, sha256 order), one trial: ~$1.6 per
            # model.  TAU2_FULL=1 runs every base task, 4 trials (~$70 per model).
            list=$BENCH_DIR/tasks/tau2-budget.json
            [ $pilot = 1 ] && list=$BENCH_DIR/tasks/tau2-pilot.json
            if [ $pilot = 1 ] || [ -n "$limit" ] || [ -z "${TAU2_FULL:-}" ]; then
                ids=(--task-ids $(python3 -c "import json,sys; ids=json.load(open(sys.argv[1]))[sys.argv[2]]; n=sys.argv[3]; print(' '.join(ids[:int(n)] if n else ids))" \
                    "$list" "$domain" "$limit"))
            fi
            [ -n "${TAU2_TASK_IDS:-}" ] && ids=(--task-ids $TAU2_TASK_IDS)   # explicit ids, e.g. for tests
            # tau2_run.py: tau2's CLI with its NL-assertion judge (retail) via OpenRouter.
            (cd "$WORK/tau2-bench" && "$WORK/tau2-venv/bin/python" "$BENCH_DIR/tau2_run.py" run --domain "$domain" \
                --agent-llm "openai/$model" --agent-llm-args "$agent_args" \
                --user-llm "$user_llm" --user-llm-args '{"temperature":0.0}' \
                --task-split-name base ${ids[@]+"${ids[@]}"} ${retrieval[@]+"${retrieval[@]}"} --num-trials "$attempts" \
                --max-concurrency "$conc" --save-to "$model-$domain-$(basename "$run")") \
                2>&1 | tee -a "$run/harness.log"
            cp -r "$WORK/tau2-bench/data/simulations/$model-$domain-$(basename "$run")" "$run/tau2-$domain"
        done
        # tau2 leaves a simulation that died on an engine error (context
        # exhausted, say) without a reward and out of its average; here it is
        # a failed task: score.txt reports the mean over every task.
        python3 - "$run" <<'PY' | tee "$run/score.txt"
import glob, json, os, sys, collections
for f in sorted(glob.glob(os.path.join(sys.argv[1], "tau2-*", "results.json"))):
    sims = json.load(open(f)).get("simulations") or []
    r = [(s.get("reward_info") or {}).get("reward") for s in sims]
    scored = [x for x in r if x is not None]
    terms = collections.Counter(s.get("termination_reason") or "unknown" for s in sims)
    print(f"{os.path.basename(os.path.dirname(f))}: mean {sum(scored) / len(r):.3f} over {len(r)} tasks"
          f" ({len(r) - len(scored)} unscored counted as 0; mean of scored {sum(scored) / max(1, len(scored)):.3f});"
          f" terminations {dict(terms)}" if r else f"{f}: no simulations")
PY
        ;;
esac

"$BENCH_DIR/serve.sh" stop
trap - EXIT
for f in "$run/requests.jsonl" "$run/ds4-trace.txt"; do
    [ -f "$f" ] && zstd -q --long=27 -19 --rm "$f"
done
python3 - "$run/manifest.json" <<'EOF'
import json, sys, time
m = json.load(open(sys.argv[1])); m["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(m, open(sys.argv[1], "w"), indent=1)
EOF
or_end=$(or_usage)
if [ -n "$or_start" ] && [ -n "$or_end" ]; then
    python3 -c 'import sys; print(f"openrouter: ${float(sys.argv[2]) - float(sys.argv[1]):.2f} this run, ${float(sys.argv[2]):.2f} on the key")' \
        "$or_start" "$or_end" | tee -a "$run/harness.log"
fi
echo "done: $run"
[ -n "${RCLONE_DEST:-}" ] && "$BENCH_DIR/sync.sh" once || true
