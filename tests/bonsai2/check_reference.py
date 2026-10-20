#!/usr/bin/env python3
"""Ternary Bonsai 2 end-to-end check against the PrismML reference.

Runs ./ds4 greedily on every case of tests/bonsai2/golden.json (rendered
prompts taken from PrismML's llama.cpp fork, see docs/BONSAI2_27B.md) and
requires the prompt tokenization and the generated token ids to match
exactly, and the top-5 log-probabilities of every step to agree within a
tolerance.  Exits non-zero on the first kind of mismatch it reports.

Usage: tests/bonsai2/check_reference.py [--model GGUF] [--ds4 ./ds4]
                                        [--tol 0.05] [--case ID ...]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_MODEL = os.path.join(ROOT, "gguf", "ternary-bonsai-2-27b", "Ternary-Bonsai-2-27B-PQ2_0.gguf")


def run_case(args, case, tmp):
    prompt = os.path.join(tmp, case["id"] + ".txt")
    out = os.path.join(tmp, case["id"] + ".json")
    with open(prompt, "w", encoding="utf-8") as f:
        f.write(case["rendered_prompt"])
    cmd = [args.ds4, "-m", args.model, "--prompt-file", prompt, "--temp", "0",
           "-n", str(len(case["generated_ids"])), "-c", "8192",
           "--dump-logprobs", out, "--logprobs-top-k", "20"]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        raise SystemExit(f"{case['id']}: ds4 exited with {r.returncode}")
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def check(case, got, tol):
    errors = []
    if got["prompt_tokens"] != len(case["prompt_ids"]):
        errors.append(f"prompt is {got['prompt_tokens']} tokens, reference {len(case['prompt_ids'])}")
    ids = [s["selected"]["id"] for s in got["steps"]]
    want = case["generated_ids"]
    first = next((i for i, (a, b) in enumerate(zip(ids, want)) if a != b), None)
    if first is None and len(ids) < len(want):
        first = len(ids)
    if first is not None:
        errors.append(f"greedy tokens diverge at step {first}")
    worst = 0.0
    for i, top in enumerate(case["top5"][:len(got["steps"]) if first is None else first]):
        lp = {t["token"]["id"]: t["logprob"] for t in got["steps"][i]["top_logprobs"]}
        for tok, ref in top:
            if tok not in lp:
                errors.append(f"step {i}: reference top-5 token {tok} missing from ds4 top-20")
                continue
            worst = max(worst, abs(lp[tok] - ref))
    if worst > tol:
        errors.append(f"top-5 logprob drift {worst:.4f} > {tol}")
    return errors, worst, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ds4", default=os.path.join(ROOT, "ds4"))
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--case", action="append")
    args = ap.parse_args()
    with open(os.path.join(HERE, "golden.json"), encoding="utf-8") as f:
        golden = json.load(f)
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for case in golden["cases"]:
            if args.case and case["id"] not in args.case:
                continue
            errors, worst, n = check(case, run_case(args, case, tmp), args.tol)
            status = "ok" if not errors else "FAIL"
            print(f"  {case['id']:<20} {n:4d} tokens  max top-5 logprob diff {worst:.4f}  {status}")
            for e in errors:
                print(f"    {e}")
            failed += bool(errors)
    print("bonsai2 reference: " + ("all cases match" if not failed else f"{failed} case(s) failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
