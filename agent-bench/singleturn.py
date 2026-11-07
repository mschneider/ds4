"""Shared pieces for the single-turn evals (forecastbench.py has its own copy):
a chat call against the proxy, a resumable concurrent runner, and an LLM judge
through OpenRouter."""
import concurrent.futures as cf, json, os, threading, time, urllib.request


def chat(api, model, messages, sampling, max_tokens, timeout=3600, key=None, tries=3):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, **sampling}
    err = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                api.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + (key or os.environ.get("OPENAI_API_KEY", "sk-local"))})
            return json.load(urllib.request.urlopen(req, timeout=timeout)), None
        except Exception as e:  # network, 5xx, timeouts: retry, then record
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    return None, err


def run_items(items, out_path, fn, concurrency, label):
    """items: list of (key, payload); fn(payload) -> dict.  Appends one JSON
    line per item to out_path and skips keys already there (resume)."""
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["key"] for l in open(out_path)}
    todo = [(k, p) for k, p in items if k not in done]
    print(f"{label}: {len(todo)} to run ({len(done)} done)", flush=True)
    lock = threading.Lock()

    def one(kp):
        k, p = kp
        t0 = time.time()
        rec = fn(p)
        rec["key"], rec["seconds"] = k, round(time.time() - t0, 1)
        with lock, open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    with cf.ThreadPoolExecutor(concurrency) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 10 == 0 or i == len(todo):
                print(f"{label}: {i}/{len(todo)}", flush=True)


def message_text(resp):
    """Final answer text (content), without the reasoning channel."""
    if not resp:
        return None
    return resp.get("choices", [{}])[0].get("message", {}).get("content")


def judge(system, user, model, max_tokens=4096):
    """One judge call through OpenRouter; model like openrouter/openai/gpt-oss-120b."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("judge: set OPENROUTER_API_KEY")
    model = model.split("/", 1)[1] if model.startswith("openrouter/") else model
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    resp, err = chat("https://openrouter.ai/api/v1", model, msgs, {"temperature": 0}, max_tokens, 600, key)
    return message_text(resp), err
