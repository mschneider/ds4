"""sha256 over a directory tree: every file's relative path and content, sorted.
Used to pin a downloaded task set: agent-bench/treehash.py DIR"""
import hashlib, os, sys

root = sys.argv[1]
h = hashlib.sha256()
for dp, dn, fn in os.walk(root):
    dn.sort()
    for f in sorted(fn):
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, root)
        h.update(rel.encode() + b"\0" + hashlib.sha256(open(p, "rb").read()).digest())
print(h.hexdigest())
