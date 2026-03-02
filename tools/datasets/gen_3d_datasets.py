#!/usr/bin/env python3
"""
tools/datasets/gen_3d_datasets.py
----------------------------------
Generalized 3D dataset generator for RSMI3D benchmarking.

Supports any N for dense (Cartesian product grid) or sparse (uniform random
sample from [0,999]^3). Outputs are compatible with bench_rsmi3d_true3d.

Outputs:
  <out_prefix>_points.raw   — raw float64 triples (x, y, z) in lex order
  <out_prefix>_offsets.bin  — raw uint64 offsets (== lex rank, 0..N-1)

Then convert to TPIE:
  build/bin/raw_to_tpie <out_prefix>_points.raw <out_prefix>_points.tpie

Usage examples:
  python3 tools/datasets/gen_3d_datasets.py \\
      --type dense --N 10000 \\
      --out_prefix build/data/dense3d_10k --verify

  python3 tools/datasets/gen_3d_datasets.py \\
      --type dense --N 100000 \\
      --out_prefix build/data/dense3d_100k --verify

  python3 tools/datasets/gen_3d_datasets.py \\
      --type sparse --N 100000 --seed 42 \\
      --out_prefix build/data/sparse3d_100k --verify
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Dense factorizations
# Hard-code nice factorizations for common sizes; fall back to auto.
# ---------------------------------------------------------------------------

_KNOWN_DIMS: dict[int, tuple[int, int, int]] = {
    10_000:    (10, 10, 100),
    100_000:   (50, 50, 40),
    1_000_000: (100, 100, 100),
}


def _factorize_dense(n: int) -> tuple[int, int, int]:
    """
    Return (d0, d1, d2) with d0*d1*d2 == n, preferring near-cubic splits.
    Raises ValueError if n has no non-trivial factorization.
    """
    if n in _KNOWN_DIMS:
        return _KNOWN_DIMS[n]

    best: tuple[int, int, int] | None = None
    best_score = float("inf")

    # Enumerate d0 up to cube root
    d0 = 1
    while d0 * d0 * d0 <= n:
        if n % d0 == 0:
            rem = n // d0
            d1 = 1
            while d1 * d1 <= rem:
                if rem % d1 == 0:
                    d2 = rem // d1
                    # score = deviation from cube
                    score = (d2 - d0) ** 2 + (d1 - d0) ** 2
                    if best is None or score < best_score:
                        best = (d0, d1, d2)
                        best_score = score
                d1 += 1
        d0 += 1

    if best is None:
        raise ValueError(f"Cannot factorize {n} into (d0,d1,d2)")
    return best


# ---------------------------------------------------------------------------
# Dense generation
# ---------------------------------------------------------------------------

def gen_dense(n: int, dims: tuple[int, int, int] | None = None
              ) -> tuple[bytes, bytes, tuple[int, int, int]]:
    """
    Generate a full Cartesian product grid of size D0*D1*D2 = n.
    Returns (pts_bytes, offsets_bytes, (D0, D1, D2)).
    """
    d0, d1, d2 = dims if dims is not None else _factorize_dense(n)
    actual_n = d0 * d1 * d2
    if actual_n != n:
        raise ValueError(
            f"--dims {d0},{d1},{d2} gives {actual_n} points, not {n}")

    pts_buf = bytearray(n * 3 * 8)
    off_buf = bytearray(n * 8)
    pv = memoryview(pts_buf).cast("d")
    ov = memoryview(off_buf).cast("Q")

    idx = 0
    for s in range(d0):
        for h1 in range(d1):
            for h2 in range(d2):
                pv[idx * 3    ] = float(s)
                pv[idx * 3 + 1] = float(h1)
                pv[idx * 3 + 2] = float(h2)
                ov[idx]          = idx
                idx += 1

    return bytes(pts_buf), bytes(off_buf), (d0, d1, d2)


# ---------------------------------------------------------------------------
# Sparse generation
# ---------------------------------------------------------------------------

def gen_sparse(n: int, seed: int) -> tuple[bytes, bytes]:
    """
    Sample n unique triples from [0,999]^3, sort lexicographically,
    assign offset = lex rank.
    The universe has 10^9 elements; encodes as key = s*1_000_000 + h1*1000 + h2.
    """
    print(f"  Sampling {n:,} unique triples from [0,999]^3 (seed={seed}) ...")
    rng = random.Random(seed)
    keys = sorted(rng.sample(range(10 ** 9), n))

    pts_buf = bytearray(n * 3 * 8)
    off_buf = bytearray(n * 8)
    pv = memoryview(pts_buf).cast("d")
    ov = memoryview(off_buf).cast("Q")

    for i, k in enumerate(keys):
        s  = k // 1_000_000
        h1 = (k // 1_000) % 1_000
        h2 = k % 1_000
        pv[i * 3    ] = float(s)
        pv[i * 3 + 1] = float(h1)
        pv[i * 3 + 2] = float(h2)
        ov[i]          = i

    return bytes(pts_buf), bytes(off_buf)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(pts_path: Path, off_path: Path, n: int) -> bool:
    ok = True

    pts_data = pts_path.read_bytes()
    if len(pts_data) != n * 3 * 8:
        print(f"  [FAIL] points size {len(pts_data)} != {n*3*8}", file=sys.stderr)
        ok = False
    else:
        pv = memoryview(pts_data).cast("d")
        min_s = min_h1 = min_h2 = float("inf")
        max_s = max_h1 = max_h2 = float("-inf")
        for i in range(n):
            s, h1, h2 = pv[i*3], pv[i*3+1], pv[i*3+2]
            min_s  = min(min_s,  s);  max_s  = max(max_s,  s)
            min_h1 = min(min_h1, h1); max_h1 = max(max_h1, h1)
            min_h2 = min(min_h2, h2); max_h2 = max(max_h2, h2)
        print(f"  count    : {n:,}")
        print(f"  SourceID : [{int(min_s)}, {int(max_s)}]")
        print(f"  Hop1_ID  : [{int(min_h1)}, {int(max_h1)}]")
        print(f"  Hop2_ID  : [{int(min_h2)}, {int(max_h2)}]")

    off_data = off_path.read_bytes()
    if len(off_data) != n * 8:
        print(f"  [FAIL] offsets size {len(off_data)} != {n*8}", file=sys.stderr)
        ok = False
    else:
        ov = memoryview(off_data).cast("Q")
        for i in range(n):
            if ov[i] != i:
                print(f"  [FAIL] offset[{i}]={ov[i]}", file=sys.stderr)
                ok = False
                break
        if ok:
            print(f"  offsets  : [0, {n-1}]  (monotonic)")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Dense or Sparse 3D datasets for RSMI3D benchmarking.")
    parser.add_argument("--type",       choices=["dense", "sparse"], required=True)
    parser.add_argument("--N",          type=int, required=True, help="Number of points")
    parser.add_argument("--seed",       type=int, default=42,
                        help="RNG seed (sparse only; default 42)")
    parser.add_argument("--out_prefix", required=True,
                        help="Output prefix, e.g. build/data/dense3d_100k")
    parser.add_argument("--dims",       type=str, default=None,
                        help="Dense only: D0,D1,D2 (default: auto-factorize)")
    parser.add_argument("--verify",     action="store_true")
    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pts_path = Path(str(out_prefix) + "_points.raw")
    off_path = Path(str(out_prefix) + "_offsets.bin")

    n = args.N

    if args.type == "dense":
        dims = None
        if args.dims:
            parts = [int(x) for x in args.dims.split(",")]
            if len(parts) != 3:
                parser.error("--dims must be D0,D1,D2")
            dims = tuple(parts)  # type: ignore[assignment]

        print(f"\n=== Dense3D  (N={n:,}) ===")
        pts, offs, (d0, d1, d2) = gen_dense(n, dims)
        print(f"  Grid     : {d0} x {d1} x {d2} = {d0*d1*d2:,}")

    else:  # sparse
        print(f"\n=== Sparse3D  (N={n:,}, seed={args.seed}) ===")
        pts, offs = gen_sparse(n, args.seed)

    pts_path.write_bytes(pts)
    off_path.write_bytes(offs)
    print(f"  wrote {pts_path}  ({len(pts)/1024/1024:.2f} MB)")
    print(f"  wrote {off_path}  ({len(offs)/1024/1024:.2f} MB)")

    if args.verify:
        print("  --- validation ---")
        if validate(pts_path, off_path, n):
            print("  validation: PASS")
        else:
            print("  validation: FAIL", file=sys.stderr)
            sys.exit(1)

    tpie_path = str(out_prefix) + "_points.tpie"
    print(f"\n  Next: convert .raw -> .tpie:")
    print(f"    build/bin/raw_to_tpie {pts_path} {tpie_path}")
    print("Done.")


if __name__ == "__main__":
    main()
