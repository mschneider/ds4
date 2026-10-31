#!/usr/bin/env python3
"""Requantize a Qwen3.8-27B "qwen35" GGUF into the ds4 4-bit recipe.

The source is llama.cpp's BF16 conversion (ggml-org/Qwen3.8-27B-GGUF), so
names, layouts, the tiled V-head order and the folded norms are already
llama.cpp's; this tool only changes tensor types:

  projections (GDN, attention, FFN) and output   --proj  (default Q4_K)
  token_embd (a row gather, read once per token) --embd  (default Q4_K)
  GDN ssm_alpha / ssm_beta (5120 x 48)           --gates (default BF16)
  norms, conv, ssm_a, dt bias                    kept (F32)

Weights already in the target type are copied byte for byte; the rest are
read as BF16/F16/F32 and quantized in row chunks with the shared native
quantizer (make -C gguf-tools quants-shared), so memory stays bounded.

  python3 gguf-tools/qwen38_27b_quantize.py \\
      --src gguf/qwen38-27b/Qwen3.8-27B-BF16.gguf \\
      --out gguf/qwen38-27b/Qwen3.8-27B-Q4K.gguf
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

from qwen4_pack import GGMLQuantizer, default_quantizer_library
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, tnbytes, w_str

T_F32, T_F16, T_BF16 = 0, 1, 30
QTYPES = {"Q4_0": 2, "Q8_0": 8, "Q4_K": 12, "BF16": T_BF16, "F16": T_F16, "F32": T_F32}
ALIGN = 32
CHUNK_ROWS = 8192

PROJECTIONS = ("attn_qkv", "attn_gate", "ssm_out", "attn_q", "attn_k", "attn_v",
               "attn_output", "ffn_gate", "ffn_up", "ffn_down")


def target_type(name: str, src_type: int, args) -> int:
    base = name.split(".", 2)[-1] if name.startswith("blk.") else name
    if name == "token_embd.weight":
        return QTYPES[args.embd]
    if name == "output.weight" or base in (p + ".weight" for p in PROJECTIONS):
        return QTYPES[args.proj]
    if base in ("ssm_alpha.weight", "ssm_beta.weight"):
        return QTYPES[args.gates]
    return src_type


def nbytes(t: int, dims) -> int:
    n = 1
    for d in dims:
        n *= d
    if t == 12:
        return n // 256 * 144
    return tnbytes(t, n)


def float_rows(raw: bytes, t: int, width: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8)
    if t == T_F32:
        v = a.view("<f4")
    elif t == T_F16:
        v = a.view("<f2").astype(np.float32)
    elif t == T_BF16:
        v = (a.view("<u2").astype(np.uint32) << 16).view("<f4")
    else:
        raise SystemExit(f"cannot requantize from type {t}; use a BF16/F16/F32 source")
    return v.reshape(-1, width)


def encode(rows: np.ndarray, t: int, q: GGMLQuantizer) -> bytes:
    if t == T_F32:
        return np.ascontiguousarray(rows, dtype="<f4").tobytes()
    if t == T_F16:
        return np.ascontiguousarray(rows, dtype="<f2").tobytes()
    if t == T_BF16:
        u = np.ascontiguousarray(rows, dtype="<f4").view("<u4")
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()
    name = next(k for k, v in QTYPES.items() if v == t)
    return q.encode(rows, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="qwen35 GGUF (BF16 for requantized tensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--proj", default="Q4_K", choices=QTYPES)
    ap.add_argument("--embd", default="Q4_K", choices=QTYPES)
    ap.add_argument("--gates", default="BF16", choices=QTYPES)
    ap.add_argument("--quants-lib", default=str(default_quantizer_library()))
    args = ap.parse_args()

    src = Reader(args.src)
    if src.kv.get("general.architecture") != "qwen35":
        raise SystemExit("source is not a qwen35 GGUF")
    q = GGMLQuantizer(args.quants_lib)

    plan = []
    for name, (t, dims, off) in src.tensors.items():
        tt = target_type(name, t, args)
        plan.append((name, t, tt, dims, off))

    kv = dict(src.kv)
    kv["ds4.qwen38_27b.recipe"] = f"proj={args.proj} embd={args.embd} gates={args.gates}"
    kv_types = dict(src.kv_types)
    kv_types["ds4.qwen38_27b.recipe"] = 8
    kv.pop("general.alignment", None)

    header = bytearray(b"GGUF" + struct.pack("<I", 3) + struct.pack("<q", len(plan)) +
                       struct.pack("<q", len(kv)))
    for k, v in kv.items():
        header += kv_bytes(k, kv_types[k], v)
    off = 0
    for name, _, tt, dims, _ in plan:
        header += w_str(name) + struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<q", d) for d in dims)
        header += struct.pack("<I", tt) + struct.pack("<Q", off)
        off = (off + nbytes(tt, dims) + ALIGN - 1) & ~(ALIGN - 1)
    header += b"\0" * ((-len(header)) % ALIGN)

    out_path = Path(args.out)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    t0 = time.time()
    with open(tmp, "wb") as out:
        out.write(header)
        for i, (name, t, tt, dims, toff) in enumerate(plan):
            width = dims[0]
            rows = nbytes(t, dims) // nbytes(t, [width])
            src_row = nbytes(t, [width])
            src.f.seek(src.data_start + toff)
            written = 0
            for r0 in range(0, rows, CHUNK_ROWS):
                n = min(CHUNK_ROWS, rows - r0)
                raw = src.f.read(n * src_row)
                blob = raw if tt == t else encode(float_rows(raw, t, width), tt, q)
                out.write(blob)
                written += len(blob)
            assert written == nbytes(tt, dims), name
            out.write(b"\0" * ((-written) % ALIGN))
            if tt != t or i % 64 == 0:
                print(f"[{i + 1}/{len(plan)}] {name} {dims} {t}->{tt} "
                      f"{time.time() - t0:.0f}s", flush=True)
    tmp.rename(out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
