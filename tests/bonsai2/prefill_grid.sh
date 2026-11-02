#!/bin/bash
# Whole-model prefill grid for Ternary Bonsai 2: every configuration on every
# prompt length, repeated in interleaved order, as CSV on stdout.
#
#   tests/bonsai2/prefill_grid.sh [-m MODEL] [-r REPEATS] [-p "15 50 100 400"] \
#       "DS4_BONSAI2_UPSTREAM_MIN=off" "DS4_BONSAI2_UPSTREAM_MIN=512" ...
#
# Each configuration is a space-separated list of VAR=VALUE settings ("" or
# "X=1" for the defaults).  -p lists prompt sizes in repeats of a 10-token
# sentence (15 ~ 170 tokens, 400 ~ 4069, 1600 ~ 16k).  Run from the
# repository root: ds4 reads metal/*.metal relative to the working directory.
#
# Timed runs need AC power and, on laptops with power modes, High Power Mode
# (pmset powermode 2); the script stops otherwise.  BONSAI2_GRID_NO_POWER=1
# skips the check (results are then not comparable).  Greedy output is
# printed per run: every configuration must produce the same text.
set -u
MODEL=gguf/ternary-bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf
REPEATS=2
SIZES="15 50 100 400"
while getopts "m:r:p:" opt; do
    case $opt in
        m) MODEL=$OPTARG ;;
        r) REPEATS=$OPTARG ;;
        p) SIZES=$OPTARG ;;
        *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || set -- "X=1"
[ -x ./ds4 ] || { echo "run from the repository root after make" >&2; exit 2; }

power_ok() {
    [ "${BONSAI2_GRID_NO_POWER:-0}" = 1 ] && return 0
    pmset -g batt | head -1 | grep -q "AC Power" || { echo "not on AC power" >&2; return 1; }
    local mode
    mode=$(pmset -g | awk '/powermode/ {print $2}')
    [ -z "$mode" ] || [ "$mode" = 2 ] || { echo "powermode is $mode, not 2 (High Power)" >&2; return 1; }
}

echo "prompt_reps,config,prefill_tps,output"
for r in $(seq 1 "$REPEATS"); do
    for reps in $SIZES; do
        prompt=$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*$reps + 'How many times did the fox jump? Answer briefly.')")
        for cfg in "$@"; do
            power_ok || exit 1
            while :; do
                # shellcheck disable=SC2086
                out=$(env $cfg ./ds4 -m "$MODEL" -c 20000 -n 4 --temp 0 -p "$prompt" 2>&1)
                grep -q "already running" <<<"$out" && { sleep 10; continue; }
                break
            done
            tps=$(grep -o 'prefill: [0-9.]*' <<<"$out" | awk '{print $2}')
            text=$(grep -v '^ds4' <<<"$out" | tail -1 | cut -c1-40 | tr ',' ' ')
            echo "$reps,$cfg,${tps:-FAILED},$text"
        done
    done
done
