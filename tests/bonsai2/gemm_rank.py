#!/usr/bin/env python3
"""Rank Bonsai 2 prefill GEMM configurations from `test_kernels --bench-gemm` CSV.

Each configuration's per-shape times are weighted by how often the shape runs
in one prefill pass of Ternary Bonsai 2 27B (64 layers: 48 Gated DeltaNet and
16 attention), so the ranking is whole-model GEMM time per token count.

    tests/bonsai2/test_kernels --bench-gemm "" 16,64,256,512,2048,4096 > gemm.csv
    tests/bonsai2/gemm_rank.py gemm.csv [more.csv ...]

For each token count it prints the best configurations against the current
production path (prod_fused / prod_unpack rows, measured with the defaults or
DS4_BONSAI2_* overrides in effect), and the per-padded-token cost of the best
narrow (BN 32) and wide (BN 64) tiles, which is what DS4_BONSAI2_GEMM_AUTO's
ratio expresses.
"""
import collections
import csv
import math
import sys

# Shape name in bench_gemm -> calls per prefill pass.
WEIGHTS = {
    "ffn_gate_up": 128,  # gate and up, every layer
    "ffn_down": 64,
    "lin_qkv": 48,
    "lin_z": 48,
    "out_6144": 64,      # GDN ssm_out and attention output
    "attn_q": 16,        # q and gate
    "attn_kv": 32,       # k and v
}


def main(paths):
    times = collections.defaultdict(dict)
    for path in paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                times[(int(row["n_tok"]), row["cfg"])][row["shape"]] = float(row["ms"])
    totals = collections.defaultdict(dict)
    for (n_tok, cfg), shapes in times.items():
        if all(s in shapes for s in WEIGHTS):
            totals[n_tok][cfg] = sum(WEIGHTS[s] * shapes[s] for s in WEIGHTS)
    for n_tok in sorted(totals):
        ranked = sorted((ms, cfg) for cfg, ms in totals[n_tok].items())
        prod = min((ms for ms, cfg in ranked if cfg.startswith("prod")), default=None)
        print(f"n_tok {n_tok}" + (f": production {prod:.1f} ms" if prod else ""))
        for ms, cfg in ranked[:6]:
            gain = f"  {100.0 * (prod / ms - 1.0):+5.1f}%" if prod else ""
            print(f"   {cfg:24s} {ms:10.1f} ms{gain}")
        per_pad = {}
        for ms, cfg in ranked:
            if cfg.startswith("prod"):
                continue
            bn = int(cfg.split("_")[0].split("x")[1])
            if bn in (32, 64) and bn not in per_pad:
                per_pad[bn] = (ms / (math.ceil(n_tok / bn) * bn), cfg)
        if 32 in per_pad and 64 in per_pad:
            print(f"   per padded token: {per_pad[32][1]} {per_pad[32][0]:.3f}, "
                  f"{per_pad[64][1]} {per_pad[64][0]:.3f}, ratio {per_pad[64][0] / per_pad[32][0]:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
