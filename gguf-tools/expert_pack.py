#!/usr/bin/env python3
"""Write MODEL.gguf.experts: the model's routed experts, one record per expert.

SSD streaming reads a missing expert's gate, up and down rows from three
tensors far apart in the GGUF, so each expert costs three SSD requests. In
the pack, each layer's experts are records of gate|up|down next to each other,
aligned to 16 KiB, and ds4 reads a whole expert with one request. The model
file is unchanged and still required; ds4 uses the pack when it sits next to
the model (or DS4_EXPERT_PACK names it) and checks it against the model.

Usage: gguf-tools/expert_pack.py MODEL.gguf [--output PATH]

Format (little endian): header {magic "DS4XPAK1", u32 version 1, u32 n,
u64 model_size, u64 pack_size, u64 header_bytes}, then n entries
{u64 model_off, u64 expert_bytes, u64 pack_base, u64 stride, u32 n_expert,
u32 0} sorted by model_off; expert e of a tensor is at
pack_base + e * stride. The ds4 reader is ds4_expert_pack.inc.
"""
import argparse
import fcntl
import os
import re
import struct
import sys

MAGIC = b'DS4XPAK1'
VERSION = 1
ALIGN = 16384
CHUNK_EXPERTS = 32
NAME = re.compile(r'^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$')
GGUF_STRING, GGUF_ARRAY = 8, 9
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
# ggml type -> (bytes per block, elements per block)
BLOCKS = {
    0: (4, 1), 1: (2, 1), 30: (2, 1),                 # F32, F16, BF16
    2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32), 8: (34, 32),
    10: (84, 256), 11: (110, 256), 12: (144, 256), 13: (176, 256), 14: (210, 256),
    16: (66, 256), 17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32),
    21: (110, 256), 22: (82, 256), 23: (136, 256), 29: (56, 256), 39: (17, 32),
}


def read_tensor_table(f):
    if f.read(4) != b'GGUF':
        raise SystemExit('not a GGUF file')
    version, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))
    if version < 2:
        raise SystemExit(f'GGUF version {version} is not supported')

    def string():
        n, = struct.unpack('<Q', f.read(8))
        return f.read(n).decode()

    def skip(t):
        if t == GGUF_STRING:
            string()
        elif t == GGUF_ARRAY:
            et, n = struct.unpack('<IQ', f.read(12))
            if et == GGUF_STRING:
                for _ in range(n):
                    string()
            else:
                f.seek(SCALAR[et] * n, 1)
        else:
            f.seek(SCALAR[t], 1)

    align = 32
    for _ in range(n_kv):
        key = string()
        t, = struct.unpack('<I', f.read(4))
        if key == 'general.alignment' and t == 4:
            align, = struct.unpack('<I', f.read(4))
        else:
            skip(t)
    tensors = []
    for _ in range(n_tensors):
        name = string()
        nd, = struct.unpack('<I', f.read(4))
        dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
        t, off = struct.unpack('<IQ', f.read(12))
        tensors.append((name, dims, t, off))
    data_start = (f.tell() + align - 1) // align * align
    return [(n, d, t, data_start + off) for n, d, t, off in tensors]


def tensor_bytes(name, dims, t):
    if t not in BLOCKS:
        raise SystemExit(f'{name}: ggml type {t} has no known block size')
    size, block = BLOCKS[t]
    if dims[0] % block:
        raise SystemExit(f'{name}: row of {dims[0]} is not a multiple of {block}')
    n = 1
    for d in dims:
        n *= d
    return n // block * size


def plan(tensors, model_size):
    layers = {}
    for name, dims, t, off in tensors:
        m = NAME.match(name)
        if m:
            nbytes = tensor_bytes(name, dims, t)
            if off + nbytes > model_size:
                raise SystemExit(f'{name}: extends past the end of the model')
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = (name, dims[-1], nbytes, off)
    if not layers:
        raise SystemExit('the model has no ffn_{gate,up,down}_exps tensors')

    records, pos = [], 0
    for layer in sorted(layers):
        parts = layers[layer]
        if set(parts) != {'gate', 'up', 'down'}:
            raise SystemExit(f'layer {layer}: needs gate, up and down experts, has {sorted(parts)}')
        n_expert = parts['gate'][1]
        if any(p[1] != n_expert or p[2] % n_expert for p in parts.values()):
            raise SystemExit(f'layer {layer}: expert counts disagree')
        sizes = [parts[k][2] // n_expert for k in ('gate', 'up', 'down')]
        stride = (sum(sizes) + ALIGN - 1) // ALIGN * ALIGN
        records.append((layer, n_expert, stride, pos,
                        [(parts[k][3], s) for k, s in zip(('gate', 'up', 'down'), sizes)]))
        pos += n_expert * stride
    return records, pos


def no_cache(fd):
    """Copying tens of GB through the page cache would evict everything else."""
    if hasattr(fcntl, 'F_NOCACHE'):
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)


def pread_full(fd, n, off):
    out = bytearray()
    while len(out) < n:
        chunk = os.pread(fd, n - len(out), off + len(out))
        if not chunk:
            raise SystemExit(f'short read at offset {off + len(out)}')
        out += chunk
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('model')
    ap.add_argument('--output', help='default: MODEL.experts')
    a = ap.parse_args()
    out_path = a.output or a.model + '.experts'

    model_size = os.path.getsize(a.model)
    with open(a.model, 'rb') as f:
        tensors = read_tensor_table(f)
    records, data_bytes = plan(tensors, model_size)

    entries = []
    for layer, n_expert, stride, base, parts in records:
        inner = 0
        for model_off, expert_bytes in parts:
            entries.append((model_off, expert_bytes, 0, stride, n_expert, inner, base))
            inner += expert_bytes
    entries.sort()
    header_len = 40 + 40 * len(entries)
    header_bytes = (header_len + ALIGN - 1) // ALIGN * ALIGN
    pack_size = header_bytes + data_bytes
    header = MAGIC + struct.pack('<IIQQQ', VERSION, len(entries), model_size, pack_size, header_bytes)
    for model_off, expert_bytes, _, stride, n_expert, inner, base in entries:
        header += struct.pack('<QQQQII', model_off, expert_bytes, header_bytes + base + inner,
                              stride, n_expert, 0)
    header += b'\0' * (header_bytes - len(header))

    gib = 1 << 30
    print(f'{a.model}: {len(records)} layers, {len(entries)} tensors -> '
          f'{out_path} ({pack_size / gib:.2f} GiB)', flush=True)
    tmp = out_path + '.part'
    src = os.open(a.model, os.O_RDONLY)
    dst = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        no_cache(src)
        no_cache(dst)
        os.ftruncate(dst, pack_size)
        os.pwrite(dst, header, 0)
        for i, (layer, n_expert, stride, base, parts) in enumerate(records):
            for e0 in range(0, n_expert, CHUNK_EXPERTS):
                n = min(CHUNK_EXPERTS, n_expert - e0)
                slabs = [pread_full(src, s * n, off + s * e0) for off, s in parts]
                buf = bytearray(stride * n)
                for j in range(n):
                    pos = j * stride
                    for (_, s), slab in zip(parts, slabs):
                        buf[pos:pos + s] = slab[j * s:(j + 1) * s]
                        pos += s
                written = 0
                start = header_bytes + base + e0 * stride
                while written < len(buf):
                    written += os.pwrite(dst, memoryview(buf)[written:], start + written)
            print(f'[{i + 1}/{len(records)}] layer {layer}: {n_expert} experts', flush=True)
        os.fsync(dst)
    except BaseException:
        os.close(dst)
        os.unlink(tmp)
        raise
    finally:
        os.close(src)
    os.close(dst)
    os.replace(tmp, out_path)
    print(f'wrote {out_path}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
