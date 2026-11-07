#!/usr/bin/env python3
"""AA-LCR v1.1 (Artificial Analysis Long Context Reasoning): 100 questions over
document sets of 70-115K tokens, answered in one turn, graded by an LLM
equality checker with AA's v1.1 system and user prompts (from the dataset
README at REV).

  aalcr.py fetch DIR
  aalcr.py run DIR OUT --api URL --model M [--epochs N] [--limit N] [--concurrency N]
  aalcr.py judge DIR OUT [--judge MODEL]       (needs OPENROUTER_API_KEY)
  aalcr.py score OUT...

AA uses GPT-5.6 Luna (medium) as the checker; we use JUDGE_MODEL for every
quant, so scores compare quants, not leaderboard numbers.  The engine needs a
context of at least prompt + max tokens (about 140K for the longest item).
"""
import argparse, csv, hashlib, json, os, re, sys, urllib.request, zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import singleturn as st

REPO = "ArtificialAnalysis/AA-LCR"
REV = "9a77ef56b717057ade24ceab4d273712a0b4f19e"   # v1.1, 2026-09-04
FILES = {"AA-LCR_Dataset.csv": "AA-LCR_Dataset.csv",
         "AA-LCR_extracted-text.zip": "extracted_text/AA-LCR_extracted-text.zip"}

PROMPT = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

JUDGE_SYSTEM = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""

JUDGE_USER = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""

HERE = os.path.dirname(os.path.abspath(__file__))


def sha_table():
    t = {}
    for line in open(os.path.join(HERE, "tasks", "aalcr.sha256")):
        if line.strip() and not line.startswith("#"):
            h, name = line.split()
            t[name] = h
    return t


def cmd_fetch(a):
    os.makedirs(a.dir, exist_ok=True)
    want = sha_table()
    for name, path in FILES.items():
        dest = os.path.join(a.dir, name)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{path}", dest + ".tmp")
            os.replace(dest + ".tmp", dest)
        got = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        if got != want[name]:
            sys.exit(f"aalcr: {name} sha256 {got} != {want[name]}")
    # The zip stores UTF-8 names without the UTF-8 flag, so zipfile decodes
    # them as cp437; undo that, and keep every path inside the target.
    root = os.path.realpath(os.path.join(a.dir, "text"))
    with zipfile.ZipFile(os.path.join(a.dir, "AA-LCR_extracted-text.zip")) as z:
        for info in z.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                name = name.encode("cp437").decode("utf-8")
            dest = os.path.realpath(os.path.join(root, name))
            if not dest.startswith(root + os.sep):
                sys.exit(f"aalcr: unsafe path in zip: {name}")
            if info.is_dir():
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as out:
                out.write(src.read())
    print(f"aalcr: v1.1 verified in {a.dir}")


def questions(d):
    rows = list(csv.DictReader(open(os.path.join(d, "AA-LCR_Dataset.csv"), encoding="utf-8")))
    root = os.path.join(d, "text", "lcr")
    for r in rows:
        docs = []
        for fn in r["data_source_filenames"].split(";"):
            docs.append(open(os.path.join(root, r["document_category"], r["document_set_id"], fn),
                             encoding="utf-8").read())
        text = "\n\n".join(f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs))
        yield r, PROMPT.format(documents_text=text, question=r["question"])


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    qs = list(questions(a.dir))[: a.limit or None]
    sampling = json.loads(a.sampling)

    def fn(p):
        r, prompt = p
        resp, err = st.chat(a.api, a.model, [{"role": "user", "content": prompt}], sampling, a.max_tokens, a.timeout)
        msg = resp["choices"][0]["message"] if resp else {}
        return {"question_id": r["question_id"], "content": msg.get("content"),
                "reasoning": msg.get("reasoning_content"), "usage": (resp or {}).get("usage"), "error": err}

    items = [(f"{r['question_id']}#{e}", (r, p)) for e in range(a.epochs) for r, p in qs]
    st.run_items(items, os.path.join(a.out, "responses.jsonl"), fn, a.concurrency, "aalcr")


def verdict(text):
    if not text:
        return None
    m = re.search(r'"verdict"\s*:\s*"?(CORRECT|INCORRECT)', text, re.I) or re.search(r"\b(INCORRECT|CORRECT)\b", text)
    return m.group(1).upper() == "CORRECT" if m else None


def cmd_judge(a):
    rows = {r["question_id"]: r for r in csv.DictReader(open(os.path.join(a.dir, "AA-LCR_Dataset.csv"), encoding="utf-8"))}
    resp = [json.loads(l) for l in open(os.path.join(a.out, "responses.jsonl"))]

    def fn(rec):
        r = rows[rec["question_id"]]
        if not rec.get("content"):
            return {"question_id": rec["question_id"], "correct": False, "judge": None, "note": "no answer"}
        text, err = st.judge(JUDGE_SYSTEM, JUDGE_USER.format(question=r["question"], official_answer=r["answer"],
                                                             candidate_answer=rec["content"]), a.judge)
        return {"question_id": rec["question_id"], "correct": verdict(text), "judge": text, "error": err}

    st.run_items([(rec["key"], rec) for rec in resp], os.path.join(a.out, "judged.jsonl"), fn, 8, "aalcr judge")


def cmd_score(a):
    for o in a.out:
        j = [json.loads(l) for l in open(os.path.join(o, "judged.jsonl"))]
        ok = sum(1 for x in j if x["correct"])
        unk = sum(1 for x in j if x["correct"] is None)
        print(f"{o}: {ok}/{len(j)} correct ({100 * ok / max(len(j), 1):.1f}%), {unk} unparsed verdicts")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch"); f.add_argument("dir")
    r = sp.add_parser("run"); r.add_argument("dir"); r.add_argument("out")
    r.add_argument("--api", required=True); r.add_argument("--model", required=True)
    r.add_argument("--epochs", type=int, default=1); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=4); r.add_argument("--sampling", default="{}")
    r.add_argument("--max-tokens", type=int, default=16384); r.add_argument("--timeout", type=float, default=7200)
    j = sp.add_parser("judge"); j.add_argument("dir"); j.add_argument("out")
    j.add_argument("--judge", default=os.environ.get("JUDGE_MODEL", "openrouter/openai/gpt-oss-120b"))
    s = sp.add_parser("score"); s.add_argument("out", nargs="+")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "run": cmd_run, "judge": cmd_judge, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
