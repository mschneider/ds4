#!/usr/bin/env python3
"""ForecastBench subset: the question sets published after the model's cutoff.

  forecastbench.py fetch DIR                      download the pinned sets, verify sha256
  forecastbench.py run DIR OUT --api URL --model M [--concurrency N] [--limit N]
  forecastbench.py score DIR OUT... [--resolutions DIR]

Prompts are ForecastBench's own zero-shot prompts (src/llm_forecaster/prompts.py
at FB_REF), filled the way its runner fills them, with the market's freeze value
included (the variant the leaderboard produces) and "today" set to the set's
forecast due date.  Answers are parsed with ForecastBench's probability regex;
when that fails we take the last asterisk-wrapped numbers instead of calling an
extraction model, and count those cases.  Forecasts are kept, so a run can be
rescored as later resolution sets arrive (fetch with --resolutions-ref).
Scores are Brier scores on resolved items only (lower is better), which makes
paired comparisons between quants meaningful; they are not leaderboard scores.
"""
import argparse, ast, concurrent.futures as cf, hashlib, json, os, random, re, sys, threading, time
import urllib.request

FB_DATA_REPO = "forecastingresearch/forecastbench-datasets"
FB_DATA_REF = "360971a45164931d9564daf40518c8bf72514828"   # 2026-11-03
FB_REF = "e8d3c94f9b7f424bbd1205007db08cf2950a45a3"        # prompts
HERE = os.path.dirname(os.path.abspath(__file__))
SETS_FILE = os.path.join(HERE, "tasks", "forecastbench-sets.tsv")   # date, sha256 of question set

MARKET_PROMPT = """
You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Market value on {freeze_datetime}:
{freeze_datetime_value}

Today's Date: {today_date}

Question resolution date: {resolution_date}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
Do not output anything else.
Answer: {{ Insert answer here }}
"""

DATASET_PROMPT = """
You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

You're going to predict the probability of the following potential outcome "at each of the resolution dates".

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Current value on {freeze_datetime}:
{freeze_datetime_value}

Value Explanation:
{freeze_datetime_value_explanation}

Today's Date: {today_date}

Question resolution dates: {list_of_resolution_dates}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. (For example, if there are n resolution dates, you would output different *p* for each resolution date) Do not output anything else.
Answer: {{ Insert answer here }}
"""

PROB = re.compile(r"(?:\*)?(\d*\.?\d+)(?:\*)?")
STARRED = re.compile(r"\*\s*(\d*\.?\d+)\s*\*")


def sets():
    out = []
    for line in open(SETS_FILE):
        if line.strip() and not line.startswith("#"):
            d, sha = line.split()
            out.append((d, sha))
    return out


def resolution_dates(q):
    v = q["resolution_dates"]
    if v == "N/A":
        return None
    return v if isinstance(v, list) else ast.literal_eval(v)


def prompt(q, due):
    rds = resolution_dates(q)
    background = q["background"]
    if q["market_info_resolution_criteria"] != "N/A":
        background += "\n" + q["market_info_resolution_criteria"]
    p = {"background": background, "resolution_criteria": q["resolution_criteria"],
         "today_date": due, "freeze_datetime": q["freeze_datetime"],
         "freeze_datetime_value": q["freeze_datetime_value"]}
    if rds is not None:
        p["question"] = q["question"].format(forecast_due_date=due,
                                             resolution_date="each of the resolution dates provided below")
        p["freeze_datetime_value_explanation"] = q["freeze_datetime_value_explanation"]
        p["list_of_resolution_dates"] = q["resolution_dates"]
        return DATASET_PROMPT.format(**p)
    p["question"] = q["question"]
    p["resolution_date"] = q["market_info_close_datetime"]
    return MARKET_PROMPT.format(**p)


def parse(text, n):
    """ForecastBench's strict parse first; else the last n starred numbers."""
    text = (text or "").strip()
    toks = text.replace(",", " ").split()
    vals = []
    for t in toks:
        m = PROB.fullmatch(t)
        if not m or not 0 <= float(m.group(1)) <= 1:
            vals = None
            break
        vals.append(float(m.group(1)))
    if vals and len(vals) == n:
        return vals, "strict"
    starred = [float(x) for x in STARRED.findall(text) if 0 <= float(x) <= 1]
    if len(starred) >= n:
        return starred[-n:], "fallback"
    return None, "failed"


def get(url, dest):
    tmp = dest + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def cmd_fetch(a):
    os.makedirs(a.dir, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{FB_DATA_REPO}"
    for d, sha in sets():
        qf = os.path.join(a.dir, f"{d}-llm.json")
        if not os.path.exists(qf):
            get(f"{base}/{FB_DATA_REF}/datasets/question_sets/{d}-llm.json", qf)
        got = hashlib.sha256(open(qf, "rb").read()).hexdigest()
        if got != sha:
            sys.exit(f"forecastbench: {qf} sha256 {got} != {sha}")
        rf = os.path.join(a.dir, f"{d}_resolution_set.json")
        try:
            get(f"{base}/{a.resolutions_ref}/datasets/resolution_sets/{d}_resolution_set.json", rf)
        except urllib.error.HTTPError as e:
            print(f"forecastbench: no resolution set for {d} at {a.resolutions_ref[:12]} ({e.code})")
    print(f"forecastbench: {len(sets())} question sets verified in {a.dir}")


def questions(dirpath, limit):
    for d, _ in sets():
        qs = json.load(open(os.path.join(dirpath, f"{d}-llm.json")))["questions"]
        if limit:
            # Deterministic, stratified enough: order by a hash of the id.
            qs = sorted(qs, key=lambda q: hashlib.sha256(f"{d}/{q['id']}".encode()).hexdigest())[:limit]
        for q in qs:
            yield d, q


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "forecasts.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["set"], r["id"]))
    todo = [(d, q) for d, q in questions(a.dir, a.limit) if (d, q["id"]) not in done]
    sampling = json.loads(a.sampling)
    lock = threading.Lock()
    print(f"forecastbench: {len(todo)} questions to forecast ({len(done)} done)", flush=True)

    def one(item):
        d, q = item
        rds = resolution_dates(q)
        n = len(rds) if rds else 1
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt(q, d)}],
                "max_tokens": a.max_tokens, **sampling}
        t0 = time.time()
        for attempt in range(3):
            try:
                req = urllib.request.Request(a.api.rstrip("/") + "/chat/completions",
                                             data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json",
                                                      "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "sk-local")})
                resp = json.load(urllib.request.urlopen(req, timeout=a.timeout))
                break
            except Exception as e:
                err = repr(e)
                time.sleep(5 * (attempt + 1))
        else:
            resp = None
        msg = (resp or {}).get("choices", [{}])[0].get("message", {}) if resp else {}
        probs, how = parse(msg.get("content"), n)
        rec = {"set": d, "id": q["id"], "source": q["source"], "resolution_dates": rds,
               "content": msg.get("content"), "reasoning": msg.get("reasoning_content"),
               "probs": probs, "parse": how if resp else "error", "usage": (resp or {}).get("usage"),
               "seconds": round(time.time() - t0, 1)}
        if not resp:
            rec["error"] = err
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return how

    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        for i, how in enumerate(ex.map(one, todo), 1):
            if i % 25 == 0 or i == len(todo):
                print(f"forecastbench: {i}/{len(todo)}", flush=True)


def load_resolutions(dirpath):
    res = {}
    for d, _ in sets():
        rf = os.path.join(dirpath, f"{d}_resolution_set.json")
        if not os.path.exists(rf):
            continue
        for r in json.load(open(rf))["resolutions"]:
            if r["resolved"] and isinstance(r["id"], str):
                res[(d, r["id"], r["resolution_date"])] = float(r["resolved_to"])
    return res


def items(out, res):
    """(set, id, horizon) -> forecast for every resolved item; unparsed -> 0.5."""
    got, stats = {}, {"strict": 0, "fallback": 0, "failed": 0, "error": 0}
    for line in open(os.path.join(out, "forecasts.jsonl")):
        r = json.loads(line)
        stats[r["parse"]] += 1
        rds = r["resolution_dates"]
        keys = [(r["set"], r["id"], h) for h in rds] if rds else \
               [k for k in res if k[0] == r["set"] and k[1] == r["id"]]
        for i, k in enumerate(keys):
            if k in res:
                p = r["probs"][i if rds else 0] if r["probs"] else 0.5
                got[k] = p
    return got, stats


def cmd_score(a):
    res = load_resolutions(a.resolutions or a.dir)
    runs = {o: items(o, res) for o in a.out}
    common = set.intersection(*(set(g) for g, _ in runs.values()))
    print(f"resolved items forecast by every run: {len(common)}")
    for o, (got, stats) in runs.items():
        b = [(got[k] - res[k]) ** 2 for k in common]
        print(f"{o}: Brier {sum(b) / max(len(b), 1):.4f} on {len(b)} items; parse {stats}")
    if len(a.out) == 2:
        x, y = (runs[o][0] for o in a.out)
        diffs = [(x[k] - res[k]) ** 2 - (y[k] - res[k]) ** 2 for k in common]
        rng = random.Random(0)
        boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(1000))
        print(f"paired Brier difference (first - second): {sum(diffs) / len(diffs):+.4f}, "
              f"95% bootstrap [{boots[25]:+.4f}, {boots[974]:+.4f}] (items are not independent)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch"); f.add_argument("dir")
    f.add_argument("--resolutions-ref", default=FB_DATA_REF, help="dataset repo commit for resolution sets")
    r = sp.add_parser("run"); r.add_argument("dir"); r.add_argument("out")
    r.add_argument("--api", required=True); r.add_argument("--model", required=True)
    r.add_argument("--concurrency", type=int, default=8); r.add_argument("--limit", type=int, default=0,
                   help="questions per set (0 = all 500)")
    r.add_argument("--sampling", default="{}", help="JSON merged into each request (temperature, top_p, ...)")
    r.add_argument("--max-tokens", type=int, default=32768); r.add_argument("--timeout", type=float, default=3600)
    s = sp.add_parser("score"); s.add_argument("dir"); s.add_argument("out", nargs="+")
    s.add_argument("--resolutions", help="directory with newer resolution sets")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "run": cmd_run, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
