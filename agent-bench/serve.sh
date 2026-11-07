#!/usr/bin/env bash
# Serve one model behind the logging proxy:
#   agent-bench/serve.sh start MODEL RUN_DIR   engine on SERVER_PORT, proxy on PROXY_PORT
#   agent-bench/serve.sh stop
# The proxy appends every request to RUN_DIR/requests.jsonl; engine and proxy
# logs go to RUN_DIR as well.  start returns once the engine answers.
set -euo pipefail
source "$(dirname "$0")/common.sh"

PIDS="$WORK/serve.pids"

stop() {
    [ -f "$PIDS" ] || return 0
    local pids
    pids=$(cat "$PIDS")
    for pid in $pids; do kill "$pid" 2>/dev/null || true; done
    # Wait for the engine to release the GPU and the port.
    for _ in $(seq 60); do
        local alive=0
        for pid in $pids; do kill -0 "$pid" 2>/dev/null && alive=1; done
        [ $alive = 0 ] && break
        sleep 1
    done
    rm -f "$PIDS"
}

start() {
    local m=$1 run=$2 engine path
    engine=$(model_engine "$m")
    path=$(model_path "$m")
    [ -f "$path.ok" ] || die "$m: run pod_setup.sh $m first"
    [ ! -f "$PIDS" ] || die "already serving; run serve.sh stop first"
    mkdir -p "$run"
    local cmd proxy_flags=(--top-logprobs "$TOP_LOGPROBS")
    case $engine in
        llama|prism)
            local dir=$WORK/llama.cpp
            [ "$engine" = prism ] && dir=$WORK/prism-llama.cpp
            # One unified KV pool for all slots; f16 KV, matching ds4.
            cmd=("$dir/build/bin/llama-server" -m "$path" --host 127.0.0.1 --port "$SERVER_PORT"
                 -c "$CTX" -np "$PARALLEL" --kv-unified -ngl 999 -fa on --jinja
                 --cache-type-k f16 --cache-type-v f16 --metrics --no-webui) ;;
        ds4)
            # No logprobs and no /apply-template in ds4-server; its --trace
            # records the rendered prompts and outputs instead.
            cmd=("$DS4_SERVER" -m "$path" --host 127.0.0.1 --port "$SERVER_PORT"
                 --ctx "$DS4_CTX" --batched-session "$PARALLEL" -n "$MAX_OUTPUT_TOKENS"
                 --trace "$run/ds4-trace.txt")
            proxy_flags=(--top-logprobs 0 --no-render) ;;
        *) die "unknown engine $engine" ;;
    esac
    printf '%q ' "${cmd[@]}" >"$run/server.cmd"
    # ds4 loads its Metal sources relative to the working directory.
    (cd "$(dirname "${cmd[0]}")" && exec "${cmd[@]}") >"$run/server.log" 2>&1 &
    echo $! >"$PIDS"
    python3 "$BENCH_DIR/proxy.py" --upstream "http://127.0.0.1:$SERVER_PORT" \
        --listen "127.0.0.1:$PROXY_PORT" --log "$run/requests.jsonl" --timeout "$REQUEST_TIMEOUT" \
        "${proxy_flags[@]}" >"$run/proxy.log" 2>&1 &
    echo $! >>"$PIDS"
    for _ in $(seq 1800); do
        if curl -fs "http://127.0.0.1:$PROXY_PORT/v1/models" >/dev/null 2>&1; then
            echo "serving $m: proxy http://127.0.0.1:$PROXY_PORT/v1, log $run/requests.jsonl"
            return 0
        fi
        kill -0 "$(head -1 "$PIDS")" 2>/dev/null || { stop; tail -20 "$run/server.log"; die "engine exited"; }
        sleep 1
    done
    stop
    die "engine not ready after 30 minutes"
}

case ${1:-} in
    start) [ $# -eq 3 ] || die "usage: serve.sh start MODEL RUN_DIR"; start "$2" "$3" ;;
    stop) stop ;;
    *) die "usage: serve.sh start MODEL RUN_DIR | stop" ;;
esac
