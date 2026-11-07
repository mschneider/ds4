#!/usr/bin/env python3
"""Logging proxy between an agent harness and an OpenAI-compatible server.

Every chat/completions request is forwarded unchanged except that top-N
logprobs are requested, and one JSON line per call is appended to the log:
the client request, the upstream response (streamed responses as their
chunks), timing, and, when the upstream offers llama-server's
/apply-template, the rendered prompt.  Logprobs the client did not ask for
are removed before the response goes back, so the harness sees what it
would see without the proxy.  The log is what later replays teacher-force
(docs/AGENTIC_EVAL.md).  Standard library only.
"""

import argparse
import copy
import http.client
import json
import select
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOGGED = ("/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions")
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host",
       "proxy-connection", "te", "trailer", "upgrade", "content-encoding"}


class State:
    def __init__(self, args):
        self.upstream = args.upstream.rstrip("/")
        self.top = args.top_logprobs
        self.render = not args.no_render
        self.timeout = args.timeout
        self.lock = threading.Lock()
        self.log = open(args.log, "a", buffering=1, encoding="utf-8")
        self.seq = 0

    def next_id(self):
        with self.lock:
            self.seq += 1
            return self.seq

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        with self.lock:
            self.log.write(line + "\n")


def strip_logprobs(obj):
    for c in obj.get("choices") or []:
        c.pop("logprobs", None)


def client_gone(sock):
    """True once the client has closed its connection (EOF or error)."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
    except BlockingIOError:
        return False
    except OSError:
        return True


class Upstream:
    """One upstream POST on a connection we own, so that a client hang-up can
    close it: the server then cancels the generation instead of finishing a
    response nobody reads (and holding its slot while the client retries)."""

    def __init__(self, st, path, data, headers, client_sock):
        u = urllib.parse.urlsplit(st.upstream)
        self.conn = http.client.HTTPConnection(u.hostname, u.port, timeout=st.timeout)
        self.conn.request("POST", u.path + path, body=data, headers=headers)
        self.client_sock = client_sock
        self.abandoned = False
        self.finished = threading.Event()
        threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        while not self.finished.wait(0.5):
            if client_gone(self.client_sock):
                self.abandoned = True
                try:
                    self.conn.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return

    def close(self):
        self.finished.set()
        self.conn.close()


def render_prompt(st, body):
    """Rendered chat prompt from llama-server, or None where unsupported."""
    if "messages" not in body:
        return body.get("prompt")
    req = {"messages": body["messages"]}
    for k in ("tools", "chat_template_kwargs"):
        if k in body:
            req[k] = body[k]
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            st.upstream + "/apply-template", data=json.dumps(req).encode(),
            headers={"Content-Type": "application/json"}), timeout=60)
        return json.loads(r.read()).get("prompt")
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    st = None

    def log_message(self, fmt, *args):
        pass

    def forward_headers(self):
        return {k: v for k, v in self.headers.items() if k.lower() not in HOP}

    def send_plain(self, status, headers, data):
        self.send_response(status)
        for k, v in headers:
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def passthrough(self, data=None):
        req = urllib.request.Request(self.st.upstream + self.path, data=data,
                                     headers=self.forward_headers(), method=self.command)
        try:
            r = urllib.request.urlopen(req, timeout=self.st.timeout)
            self.send_plain(r.status, r.getheaders(), r.read())
        except urllib.error.HTTPError as e:
            self.send_plain(e.code, e.headers.items(), e.read())

    def do_GET(self):
        self.passthrough()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        path = self.path.split("?")[0]
        if path not in LOGGED:
            return self.passthrough(raw)
        try:
            body = json.loads(raw)
        except ValueError:
            return self.passthrough(raw)
        st = self.st
        rid = st.next_id()
        rec = {"id": rid, "path": path, "t0": time.time(), "request": body,
               "session": self.headers.get("x-session-id")}
        up = copy.deepcopy(body)
        client_wants = bool(body.get("logprobs"))
        if st.top > 0:
            if path.endswith("chat/completions"):
                up["logprobs"] = True
                up["top_logprobs"] = max(st.top, int(body.get("top_logprobs") or 0))
            else:
                up["logprobs"] = max(st.top, int(body.get("logprobs") or 0))
        stream = bool(body.get("stream"))
        u = None
        try:
            u = Upstream(st, self.path, json.dumps(up).encode(),
                         {**self.forward_headers(), "Content-Type": "application/json"},
                         self.connection)
            r = u.conn.getresponse()
            if r.status >= 400:
                data = r.read()
                u.close()
                rec.update(status=r.status, error=data.decode("utf-8", "replace"), t1=time.time())
                st.write(rec)
                return self.send_plain(r.status, r.getheaders(), data)
            if not stream:
                data = r.read()
        except Exception as e:
            if u is not None:
                u.close()
            if u is not None and u.abandoned:
                # The client gave up; nobody is left to answer.
                rec.update(status=499, error="client disconnected", t1=time.time())
                st.write(rec)
                self.close_connection = True
                return
            rec.update(status=502, error=repr(e), t1=time.time())
            st.write(rec)
            return self.send_plain(502, [("Content-Type", "application/json")],
                                   json.dumps({"error": {"message": repr(e)}}).encode())
        rec["status"] = r.status
        if not stream:
            u.close()
            try:
                resp = json.loads(data)
                rec["response"] = copy.deepcopy(resp)
                if not client_wants:
                    strip_logprobs(resp)
                    data = json.dumps(resp).encode()
            except ValueError:
                rec["response_raw"] = data.decode("utf-8", "replace")
            rec["t1"] = time.time()
            if st.render:
                rec["prompt"] = render_prompt(st, body)
            st.write(rec)
            return self.send_plain(r.status, r.getheaders(), data)

        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        chunks = []

        def emit(b):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))

        try:
            for line in r:
                s = line.strip()
                if s.startswith(b"data:") and s[5:].strip() not in (b"", b"[DONE]"):
                    try:
                        obj = json.loads(s[5:])
                        chunks.append(copy.deepcopy(obj))
                        if not client_wants:
                            strip_logprobs(obj)
                            line = b"data: " + json.dumps(obj).encode() + b"\n"
                    except ValueError:
                        pass
                emit(line)
            self.wfile.write(b"0\r\n\r\n")
        finally:
            u.close()
            if u.abandoned:
                rec["error"] = "client disconnected"
            rec.update(chunks=chunks, t1=time.time())
            if st.render:
                rec["prompt"] = render_prompt(st, body)
            st.write(rec)


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # Clients that hang up (readiness probes, cancelled agents) are routine.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upstream", required=True, help="server base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--listen", default="127.0.0.1:9000", help="host:port to serve on")
    ap.add_argument("--log", required=True, help="JSONL file to append to")
    ap.add_argument("--top-logprobs", type=int, default=20, help="0 disables logprob capture")
    ap.add_argument("--no-render", action="store_true", help="skip /apply-template prompt rendering")
    ap.add_argument("--timeout", type=float, default=14400.0, help="upstream timeout in seconds")
    args = ap.parse_args()
    Handler.st = State(args)
    host, port = args.listen.rsplit(":", 1)
    srv = Server((host, int(port)), Handler)
    srv.daemon_threads = True
    print(f"proxy: {args.listen} -> {args.upstream}, log {args.log}", file=sys.stderr)
    srv.serve_forever()


if __name__ == "__main__":
    main()
