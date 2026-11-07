#!/usr/bin/env bash
# Equivalence check before an agent run:
#   agent-bench/equiv.sh MODEL
# Scores the official continuation fixtures with this machine's engine and
# compares them with the reference scores in agent-bench/reference (ds4 on
# Metal for the 27B family, ds4 CUDA on a DGX Spark for Flash Next).  The run
# is valid when the two implementations differ far less than quants do: the
# gate is |relative NLL change| <= EQUIV_MAX_NLL_PCT (default 0.5) on both sets.
set -euo pipefail
source "$(dirname "$0")/common.sh"

[ $# -eq 1 ] || die "usage: equiv.sh MODEL"
m=$1
engine=$(model_engine "$m")
path=$(model_path "$m")
[ -f "$path.ok" ] || die "$m: run pod_setup.sh $m first"
limit=${EQUIV_MAX_NLL_PCT:-0.5}
QT="$REPO_DIR/gguf-tools/quality-testing"
out="$RESULTS_DIR/$m/equiv-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"

case $m in
    flash-next-*) sets="qwen38-flash-alibaba-100 qwen38-flash-alibaba-long" ;;
    *) sets="qwen38-27b-ionstream-fp8-100 qwen38-27b-ionstream-fp8-long" ;;
esac
for s in $sets; do
    [ -f "$QT/data/$s/manifest.tsv" ] || die "fixtures $s missing; copy gguf-tools/quality-testing/data/$s to this machine"
done

case $engine in
    llama|prism)
        dir=$WORK/llama.cpp
        [ "$engine" = prism ] && dir=$WORK/prism-llama.cpp
        make -C "$REPO_DIR/gguf-tools" -s quality-llama-score LLAMA_CPP_DIR="$dir" LLAMA_CPP_BUILD="$dir/build"
        scorer=("$QT/score_llama") ;;
    ds4)
        make -C "$REPO_DIR/gguf-tools" -s quality-score
        scorer=("$QT/score_official") ;;
esac

fail=0
for s in $sets; do
    case $s in *-long) kind=long ctx=32768 ;; *) kind=short ctx=4096 ;; esac
    ref="$BENCH_DIR/reference/$m-$kind.tsv"
    (cd "$REPO_DIR" && "${scorer[@]}" "$path" "$QT/data/$s/manifest.tsv" "$out/$kind.tsv" "$ctx") \
        >"$out/$kind.log" 2>&1 || { tail -5 "$out/$kind.log"; die "scoring $s failed"; }
    if [ ! -f "$ref" ]; then
        echo "equiv: no reference $ref yet; produce it on the reference machine (docs/AGENTIC_EVAL.md)"
        fail=1
        continue
    fi
    python3 "$QT/compare_scores.py" "$ref" "$out/$kind.tsv" | tee "$out/$kind-compare.txt"
    pct=$(awk -F'\t' '$1 == "relative_nll_change" { sub("%", "", $2); print $2 }' "$out/$kind-compare.txt")
    if awk -v p="$pct" -v l="$limit" 'BEGIN { a = p < 0 ? -p : p; exit !(a > l) }'; then
        echo "equiv: $kind FAIL, NLL differs by $pct% (limit $limit%)"
        fail=1
    else
        echo "equiv: $kind ok, NLL differs by $pct%"
    fi
done
echo "equiv: results in $out"
exit $fail
