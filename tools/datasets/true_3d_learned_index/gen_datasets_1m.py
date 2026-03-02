#!/usr/bin/env python3
"""
Generate 1M Dense 3D and 1M Sparse 3D datasets for RSMI3D benchmarking.

Dense  (100×100×100 full grid):
  SourceID  in [0, 99]  |  Hop1_ID in [0, 99]  |  Hop2_ID in [0, 99]
  Offset = lex rank = SourceID*10000 + Hop1_ID*100 + Hop2_ID

Sparse (1M unique random triples from [0,999]^3):
  Sorted lexicographically, Offset = rank after sort (0..999999)

Outputs (.raw = raw float64 triples; .bin = raw uint64 offsets):
  <out_dir>/dense3d_1m_points.raw   <out_dir>/dense3d_1m_offsets.bin
  <out_dir>/sparse3d_1m_points.raw  <out_dir>/sparse3d_1m_offsets.bin

Then convert .raw -> .tpie:
  build/bin/raw_to_tpie <name>_points.raw <name>_points.tpie

Usage:
  python3 gen_datasets_1m.py --datasets both --out_dir build/data --verify
  python3 gen_datasets_1m.py --datasets dense --out_dir build/data
  python3 gen_datasets_1m.py --datasets sparse --out_dir build/data --seed 42 --verify
"""
from __future__ import annotations

import argparse
import random
import struct
import sys
from pathlib import Path

N = 1_000_000

# ---------------------------------------------------------------------------
# Dense: 100 x 100 x 100
# ---------------------------------------------------------------------------

def gen_dense() -> tuple[bytes, bytes]:
    D0, D1, D2 = 100, 100, 100
    pts_buf = bytearray(N * 3 * 8)
    off_buf = bytearray(N * 8)
    pv = memoryview(pts_buf).cast('d')
    ov = memoryview(off_buf).cast('Q')
    idx = 0
    for s in range(D0):
        for h1 in range(D1):
            for h2 in range(D2):
                pv[idx * 3    ] = float(s)
                pv[idx * 3 + 1] = float(h1)
                pv[idx * 3 + 2] = float(h2)
                ov[idx]          = idx          # Offset == lex rank
                idx += 1
    return bytes(pts_buf), bytes(off_buf)


# ---------------------------------------------------------------------------
# Sparse: 1M unique triples from [0,999]^3
# ---------------------------------------------------------------------------

def gen_sparse(seed: int) -> tuple[bytes, bytes]:
    """
    Encodes each (s, h1, h2) as a single integer key = s*1_000_000 + h1*1000 + h2.
    random.sample(range(10**9), N) draws N distinct keys without replacement
    from the 10^9-element universe. After sorting, decoding gives lex order.
    """
    print(f"  Sampling {N:,} unique triples from [0,999]^3 (seed={seed}) ...")
    rng = random.Random(seed)
    # random.sample on a range is O(N) in CPython (reservoir/hash-set internally)
    keys = sorted(rng.sample(range(10 ** 9), N))   # sorted ints = lex order

    pts_buf = bytearray(N * 3 * 8)
    off_buf = bytearray(N * 8)
    pv = memoryview(pts_buf).cast('d')
    ov = memoryview(off_buf).cast('Q')
    for i, k in enumerate(keys):
        s   = k // 1_000_000
        h1  = (k // 1_000) % 1_000
        h2  = k % 1_000
        pv[i * 3    ] = float(s)
        pv[i * 3 + 1] = float(h1)
        pv[i * 3 + 2] = float(h2)
        ov[i]          = i
    return bytes(pts_buf), bytes(off_buf)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_points(path: Path, n: int, max_dim0: int, max_dim1: int, max_dim2: int) -> bool:
    data = path.read_bytes()
    if len(data) != n * 3 * 8:
        print(f"  [FAIL] size {len(data)} != {n*3*8}", file=sys.stderr)
        return False
    view = memoryview(data).cast('d')
    min_s = min_h1 = min_h2 = float('inf')
    max_s = max_h1 = max_h2 = float('-inf')
    for i in range(n):
        s, h1, h2 = view[i*3], view[i*3+1], view[i*3+2]
        if s  < min_s:  min_s  = s
        if s  > max_s:  max_s  = s
        if h1 < min_h1: min_h1 = h1
        if h1 > max_h1: max_h1 = h1
        if h2 < min_h2: min_h2 = h2
        if h2 > max_h2: max_h2 = h2
    print(f"  count    : {n:,}")
    print(f"  SourceID : [{int(min_s)}, {int(max_s)}]  (universe [0, {max_dim0-1}])")
    print(f"  Hop1_ID  : [{int(min_h1)}, {int(max_h1)}]  (universe [0, {max_dim1-1}])")
    print(f"  Hop2_ID  : [{int(min_h2)}, {int(max_h2)}]  (universe [0, {max_dim2-1}])")
    return True


def validate_offsets(path: Path, n: int) -> bool:
    data = path.read_bytes()
    if len(data) != n * 8:
        print(f"  [FAIL] size {len(data)} != {n*8}", file=sys.stderr)
        return False
    view = memoryview(data).cast('Q')
    for i in range(n):
        if view[i] != i:
            print(f"  [FAIL] offset[{i}] = {view[i]}", file=sys.stderr)
            return False
    print(f"  offsets  : [0, {n-1}]  (monotonic)")
    return True


def write_and_verify(name: str, pts: bytes, offs: bytes,
                     out_dir: Path, verify: bool,
                     max_dims: tuple[int, int, int]) -> None:
    pts_path = out_dir / f"{name}_points.raw"
    off_path = out_dir / f"{name}_offsets.bin"
    pts_path.write_bytes(pts)
    off_path.write_bytes(offs)
    print(f"  wrote {pts_path}  ({len(pts)/1024/1024:.1f} MB)")
    print(f"  wrote {off_path}  ({len(offs)/1024/1024:.1f} MB)")

    if verify:
        print("  --- validation ---")
        ok_p = validate_points(pts_path, N, *max_dims)
        ok_o = validate_offsets(off_path, N)
        if ok_p and ok_o:
            print("  validation: PASS")
        else:
            print("  validation: FAIL", file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate Dense3D_1M and/or Sparse3D_1M datasets.")
    p.add_argument("--datasets", choices=["dense", "sparse", "both"],
                   default="both")
    p.add_argument("--out_dir",  default="build/data")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--verify",   action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets in ("dense", "both"):
        print("\n=== Dense3D_1M  (100 x 100 x 100 full grid) ===")
        pts, offs = gen_dense()
        write_and_verify("dense3d_1m", pts, offs, out_dir, args.verify,
                         (100, 100, 100))

    if args.datasets in ("sparse", "both"):
        print(f"\n=== Sparse3D_1M  (1M from [0,999]^3, seed={args.seed}) ===")
        pts, offs = gen_sparse(args.seed)
        write_and_verify("sparse3d_1m", pts, offs, out_dir, args.verify,
                         (1000, 1000, 1000))

    print("\n--- Next: convert .raw -> .tpie ---")
    print("  build/bin/raw_to_tpie build/data/dense3d_1m_points.raw  build/data/dense3d_1m_points.tpie")
    print("  build/bin/raw_to_tpie build/data/sparse3d_1m_points.raw build/data/sparse3d_1m_points.tpie")
    print("Done.")


if __name__ == "__main__":
    main()
