#!/usr/bin/env python3
"""Model-free test of proxy.py against a fake OpenAI-compatible upstream."""

import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
seen = []
cancelled = []   # seconds the slow request ran before its client vanished


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"data": [{"id": "fake"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            return self.reply({"prompt": "<|im_start|>user\n" + body["messages"][-1]["content"]})
        seen.append(body)
        if body["messages"][-1]["content"] == "slow":
            # Generate for up to 10 s, watching the socket the way ds4-server's
            # wait_for_job_or_disconnect does.
            t0 = time.time()
            while time.time() - t0 < 10:
                r, _, _ = select.select([self.connection], [], [], 0.1)
                if r and self.connection.recv(1, socket.MSG_PEEK) == b"":
                    cancelled.append(time.time() - t0)
                    return
        lp = {"content": [{"token": "hi", "logprob": -0.1,
                           "top_logprobs": [{"token": "hi", "logprob": -0.1}]}]} \
            if body.get("logprobs") else None
        if not body.get("stream"):
            return self.reply({"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                                            "logprobs": lp, "finish_reason": "stop"}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("h", "i"):
            chunk = {"choices": [{"index": 0, "delta": {"content": piece}, "logprobs": lp}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True


def post(url, obj, timeout=10):
    r = urllib.request.urlopen(urllib.request.Request(
        url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}), timeout=timeout)
    return r.read()


def main():
    up = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    upstream = f"http://127.0.0.1:{up.server_address[1]}"
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "log.jsonl")
        port = 19000 + os.getpid() % 1000
        proc = subprocess.Popen([sys.executable, os.path.join(HERE, "proxy.py"), "--upstream", upstream,
                                 "--listen", f"127.0.0.1:{port}", "--log", log])
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(50):
                try:
                    urllib.request.urlopen(base + "/v1/models", timeout=1)
                    break
                except OSError:
                    time.sleep(0.1)
            msg = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
            plain = json.loads(post(base + "/v1/chat/completions", msg))
            assert plain["choices"][0]["message"]["content"] == "hi"
            assert "logprobs" not in plain["choices"][0], "unrequested logprobs leaked"
            assert seen[-1]["logprobs"] is True and seen[-1]["top_logprobs"] == 20
            own = json.loads(post(base + "/v1/chat/completions", {**msg, "logprobs": True, "top_logprobs": 3}))
            assert own["choices"][0]["logprobs"], "requested logprobs dropped"
            assert seen[-1]["top_logprobs"] == 20
            streamed = post(base + "/v1/chat/completions", {**msg, "stream": True}).decode()
            assert streamed.count("data:") == 3 and "logprobs" not in streamed
            time.sleep(0.3)
            recs = [json.loads(l) for l in open(log)]
            assert len(recs) == 3, recs
            assert all(r["status"] == 200 and r["prompt"].endswith("x") for r in recs)
            assert recs[0]["response"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            assert recs[0]["request"] == msg, "logged request must be the client's"
            assert len(recs[2]["chunks"]) == 2 and recs[2]["chunks"][0]["choices"][0]["logprobs"]
            # A client that gives up must cancel the upstream request too.
            try:
                post(base + "/v1/chat/completions", {**msg, "messages": [{"role": "user", "content": "slow"}]}, 1)
                raise AssertionError("slow request answered")
            except OSError:
                pass
            time.sleep(2)
            assert cancelled and cancelled[0] < 3, f"upstream not cancelled: {cancelled}"
            last = json.loads(open(log).read().splitlines()[-1])
            assert last["status"] == 499 and last["error"] == "client disconnected", last
            assert json.loads(post(base + "/v1/chat/completions", msg))["choices"], "proxy unhealthy after cancel"
        finally:
            proc.terminate()
            proc.wait()
    up.shutdown()
    print("proxy: ok")


if __name__ == "__main__":
    main()
