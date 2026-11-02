#!/usr/bin/env python3
"""Expert pack round trip: synthetic GGUF -> gguf-tools/expert_pack.py ->
tests/test_expert_pack.c (the ds4_expert_pack.inc reader).

Run from the repo root: python3 tests/test_expert_pack.py
"""
import os
import random
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gguf_str(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def write_gguf(path, tensors, align=32):
    """tensors: (name, dims, ggml type, payload bytes)."""
    kv = gguf_str('general.alignment') + struct.pack('<II', 4, align)
    infos, off = b'', 0
    offsets = []
    for name, dims, t, data in tensors:
        off = (off + align - 1) // align * align
        offsets.append(off)
        infos += gguf_str(name) + struct.pack('<I', len(dims)) + \
            struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', t, off)
        off += len(data)
    head = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), 1) + kv + infos
    data_start = (len(head) + align - 1) // align * align
    with open(path, 'wb') as f:
        f.write(head + b'\0' * (data_start - len(head)))
        for (_, _, _, data), o in zip(tensors, offsets):
            f.seek(data_start + o)
            f.write(data)
    return [data_start + o for o in offsets]


def main():
    rng = random.Random(1)
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, 'm.gguf')
        n_expert = 6
        tensors, table = [], []
        for layer in range(3):
            tensors.append((f'blk.{layer}.attn_norm.weight', (64,), 0, rng.randbytes(256)))
            # gate/up: Q8_0 rows of 64 (2 blocks = 68 bytes), 5 rows;
            # down: Q2_K row of 256 (84 bytes), 3 rows. Odd sizes on purpose.
            for kind, dims, t, eb in (('gate', (64, 5, n_expert), 8, 340),
                                      ('up', (64, 5, n_expert), 8, 340),
                                      ('down', (256, 3, n_expert), 10, 252)):
                tensors.append((f'blk.{layer}.ffn_{kind}_exps.weight', dims, t,
                                rng.randbytes(eb * n_expert)))
        # Store the tensors out of gate/up/down order to exercise sorting.
        rng.shuffle(tensors)
        offs = write_gguf(model, tensors)
        by_name = {name: (off, len(data)) for (name, _, _, data), off in zip(tensors, offs)}
        for layer in range(3):
            for kind in ('gate', 'up', 'down'):
                off, n = by_name[f'blk.{layer}.ffn_{kind}_exps.weight']
                table.append(f'{off} {n // n_expert} {n_expert}')
        table_path = os.path.join(tmp, 'table.txt')
        with open(table_path, 'w') as f:
            f.write('\n'.join(table) + '\n')

        subprocess.run([sys.executable, os.path.join(ROOT, 'gguf-tools', 'expert_pack.py'), model],
                       check=True, stdout=subprocess.DEVNULL)
        exe = os.path.join(tmp, 'test_expert_pack')
        subprocess.run(['cc', '-O1', '-Wall', '-Wextra', '-Werror', '-o', exe,
                        os.path.join(ROOT, 'tests', 'test_expert_pack.c')], check=True)
        subprocess.run([exe, model, table_path], check=True)

        # Flip the last data byte of the last expert of layer 2's down rows:
        # the reader's spot check must refuse the pack.
        pack = model + '.experts'
        off, _ = by_name['blk.2.ffn_down_exps.weight']
        with open(pack, 'rb') as f:
            n_entries, = struct.unpack('<12xI', f.read(16))
            f.seek(40)
            entries = [struct.unpack('<QQQQII', f.read(40)) for _ in range(n_entries)]
        _, expert_bytes, base, stride, n, _ = next(x for x in entries if x[0] == off)
        pos = base + (n - 1) * stride + expert_bytes - 1
        with open(pack, 'r+b') as f:
            f.seek(pos)
            b = f.read(1)
            f.seek(pos)
            f.write(bytes([b[0] ^ 0xff]))
        subprocess.run([exe, model, '--rejected'], check=True)
        os.truncate(pack, os.path.getsize(pack) - 1)
        subprocess.run([exe, model, '--rejected'], check=True)
    print('test_expert_pack: ok')


if __name__ == '__main__':
    main()
