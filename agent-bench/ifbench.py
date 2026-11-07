#!/usr/bin/env python3
"""IFBench (allenai, 299 prompts with out-of-distribution format constraints).

  ifbench.py fetch DIR
  ifbench.py run DIR OUT --api URL --model M [--epochs N] [--limit N] [--concurrency N]
  ifbench.py score DIR OUT        (runs IFBench's own run_eval in IFBENCH_VENV)

Generation is ours: one user turn per prompt, final answer text only.  Scoring
is IFBench's own verifier code (run_eval.py at REV, installed on the pod by
pod_setup.sh), one call per epoch; we report prompt-level strict accuracy
averaged over epochs, which is what Meta reports for Muse Glimmer (4 runs).
"""
import argparse, hashlib, json, os, subprocess, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import singleturn as st

REPO = "allenai/IFBench"
REV = "cd189a9c2889526d8af901d2103fd3e1ffcea6df"   # 2026-11-02
DATA_SHA = "d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a"


def cmd_fetch(a):
    os.makedirs(a.dir, exist_ok=True)
    dest = os.path.join(a.dir, "IFBench_test.jsonl")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{REPO}/{REV}/data/IFBench_test.jsonl", dest + ".tmp")
        os.replace(dest + ".tmp", dest)
    got = hashlib.sha256(open(dest, "rb").read()).hexdigest()
    if got != DATA_SHA:
        sys.exit(f"ifbench: sha256 {got} != {DATA_SHA}")
    print(f"ifbench: test set verified in {a.dir}")


def rows(d):
    return [json.loads(l) for l in open(os.path.join(d, "IFBench_test.jsonl"))]


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    data = rows(a.dir)[: a.limit or None]
    sampling = json.loads(a.sampling)

    def fn(r):
        resp, err = st.chat(a.api, a.model, [{"role": "user", "content": r["prompt"]}], sampling, a.max_tokens, a.timeout)
        msg = resp["choices"][0]["message"] if resp else {}
        return {"prompt_key": r["key"], "prompt": r["prompt"], "content": msg.get("content"),
                "reasoning": msg.get("reasoning_content"), "usage": (resp or {}).get("usage"), "error": err}

    items = [(f"{r['key']}#{e}", r) for e in range(a.epochs) for r in data]
    st.run_items(items, os.path.join(a.out, "responses.jsonl"), fn, a.concurrency, "ifbench")


def cmd_score(a):
    venv = os.environ.get("IFBENCH_VENV")
    src = os.environ.get("IFBENCH_SRC")
    if not venv or not src:
        sys.exit("ifbench: set IFBENCH_VENV and IFBENCH_SRC (pod_setup.sh installs them)")
    recs = [json.loads(l) for l in open(os.path.join(a.out, "responses.jsonl"))]
    epochs = sorted({r["key"].split("#")[1] for r in recs})
    data = [r for r in rows(a.dir) if any(x["prompt_key"] == r["key"] for x in recs)]
    data_path = os.path.join(a.out, "input_data.jsonl")
    with open(data_path, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
    accs = []
    for e in epochs:
        resp_path = os.path.join(a.out, f"responses-{e}.jsonl")
        with open(resp_path, "w") as f:
            for r in recs:
                if r["key"].endswith("#" + e):
                    f.write(json.dumps({"prompt": r["prompt"], "response": r["content"] or ""}) + "\n")
        outdir = os.path.join(a.out, f"eval-{e}")
        subprocess.run([os.path.join(venv, "bin", "python"), "-m", "run_eval", f"--input_data={data_path}",
                        f"--input_response_data={resp_path}", f"--output_dir={outdir}"], cwd=src, check=True)
        # run_eval names its output after the response file.
        strict = [json.loads(l) for l in open(os.path.join(outdir, f"responses-{e}-eval_results_strict.jsonl"))]
        acc = sum(1 for x in strict if x.get("follow_all_instructions")) / max(len(strict), 1)
        accs.append(acc)
        print(f"ifbench epoch {e}: prompt-level strict {100 * acc:.1f}% on {len(strict)} prompts")
    print(f"ifbench: mean prompt-level strict {100 * sum(accs) / len(accs):.1f}% over {len(accs)} epochs")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch"); f.add_argument("dir")
    r = sp.add_parser("run"); r.add_argument("dir"); r.add_argument("out")
    r.add_argument("--api", required=True); r.add_argument("--model", required=True)
    r.add_argument("--epochs", type=int, default=4); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=8); r.add_argument("--sampling", default="{}")
    r.add_argument("--max-tokens", type=int, default=32768); r.add_argument("--timeout", type=float, default=3600)
    s = sp.add_parser("score"); s.add_argument("dir"); s.add_argument("out")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
