#!/usr/bin/env python3
"""
Generate a 1,000,000-point dense 3D dataset for RSMI3D benchmarking.

Grid: 100 x 100 x 100  (SourceID x Hop1_ID x Hop2_ID)
  SourceID  in [0, 99]
  Hop1_ID   in [0, 99]
  Hop2_ID   in [0, 99]
  Offset    = SourceID*10000 + Hop1_ID*100 + Hop2_ID  (lex rank, 0-indexed)

Outputs (paths configurable via --points and --offsets):
  <points_out.tpie>  – TPIE file_stream<double>: 3 doubles per record (S, H1, H2)
                        written in lexicographic order (S outer, H2 inner)
  <offsets_out.bin>  – raw binary: 1M little-endian uint64_t values,
                        same row order as the points file

NOTE: TPIE file_stream<double> format:
  - 8-byte TPIE stream header (written by tpie::file_stream internally)
  - then N*3 IEEE-754 doubles, little-endian, contiguous
  Because Python can't write the TPIE header directly, this script writes:
    1. A raw binary points file  (<points_out>.raw) with N*3 doubles
    2. The offsets binary file   (<offsets_out>.bin) with N uint64_t
  Then a small C++ shim (tpie_wrap.cpp, compiled on-the-fly if tpiepy unavailable)
  can convert the raw file to TPIE format -- OR the caller can pass
  --tpie_convert to attempt direct TPIE writing via a subprocess.

  SIMPLER alternative used here: write a raw-doubles file and also write a
  helper script that calls `convert_raw_to_tpie` (the existing convert_data
  binary in build/bin) to produce the final TPIE file.  The conversion step
  is documented in the run instructions.

  If --direct_tpie is set AND the `tpie_write_stream` helper binary exists,
  it is called to produce the TPIE file directly.

Usage:
  python3 gen_true3d_1m.py                      # use default output paths
  python3 gen_true3d_1m.py --out_dir build/data
  python3 gen_true3d_1m.py --out_dir build/data --verify
  python3 gen_true3d_1m.py --dims 100 100 100 --out_dir build/data --verify
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DIM0 = 100   # SourceID
DEFAULT_DIM1 = 100   # Hop1_ID
DEFAULT_DIM2 = 100   # Hop2_ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate dense 3D dataset for RSMI3D benchmarking."
    )
    p.add_argument(
        "--dims", nargs=3, type=int,
        metavar=("DIM0", "DIM1", "DIM2"),
        default=[DEFAULT_DIM0, DEFAULT_DIM1, DEFAULT_DIM2],
        help="Grid dimensions (must multiply to desired N). Default: 100 100 100",
    )
    p.add_argument(
        "--out_dir", default="build/data",
        help="Output directory. Default: build/data",
    )
    p.add_argument(
        "--points_name", default="true3d_1m_points.raw",
        help="Filename for the raw doubles points file (before TPIE conversion).",
    )
    p.add_argument(
        "--offsets_name", default="true3d_1m_offsets.bin",
        help="Filename for the raw binary uint64_t offsets file.",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Verify outputs after writing.",
    )
    p.add_argument(
        "--tpie_binary",
        help="Path to a helper binary that converts raw doubles -> TPIE. "
             "If provided, it is called as: <binary> <raw_file> <tpie_file> <N> 3",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(dim0: int, dim1: int, dim2: int) -> Tuple[bytes, bytes]:
    """
    Returns:
      points_bytes  – N*3 doubles packed as little-endian '<d' each
      offsets_bytes – N uint64_t packed as little-endian '<Q' each
    """
    N = dim0 * dim1 * dim2
    points_buf  = bytearray(N * 3 * 8)   # 3 doubles * 8 bytes
    offsets_buf = bytearray(N * 8)        # 1 uint64_t * 8 bytes

    pts_view = memoryview(points_buf).cast('d')   # float64 view
    off_view = memoryview(offsets_buf).cast('Q')   # uint64 view

    idx = 0
    for s in range(dim0):
        for h1 in range(dim1):
            for h2 in range(dim2):
                pts_view[idx * 3    ] = float(s)
                pts_view[idx * 3 + 1] = float(h1)
                pts_view[idx * 3 + 2] = float(h2)
                off_view[idx]          = s * dim1 * dim2 + h1 * dim2 + h2
                idx += 1

    return bytes(points_buf), bytes(offsets_buf)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def verify_raw_points(raw_path: Path, dim0: int, dim1: int, dim2: int) -> bool:
    N = dim0 * dim1 * dim2
    data = raw_path.read_bytes()
    expected_bytes = N * 3 * 8
    if len(data) != expected_bytes:
        print(f"  [FAIL] points file size {len(data)} != expected {expected_bytes}", file=sys.stderr)
        return False

    view = memoryview(data).cast('d')
    errors = 0
    min_s = min_h1 = min_h2 = float('inf')
    max_s = max_h1 = max_h2 = float('-inf')
    idx = 0
    for s in range(dim0):
        for h1 in range(dim1):
            for h2 in range(dim2):
                rs  = view[idx * 3    ]
                rh1 = view[idx * 3 + 1]
                rh2 = view[idx * 3 + 2]
                if rs  < min_s:  min_s  = rs
                if rs  > max_s:  max_s  = rs
                if rh1 < min_h1: min_h1 = rh1
                if rh1 > max_h1: max_h1 = rh1
                if rh2 < min_h2: min_h2 = rh2
                if rh2 > max_h2: max_h2 = rh2
                if int(rs) != s or int(rh1) != h1 or int(rh2) != h2:
                    if errors < 5:
                        print(f"  [FAIL] row {idx}: expected ({s},{h1},{h2}), "
                              f"got ({rs},{rh1},{rh2})", file=sys.stderr)
                    errors += 1
                idx += 1

    print(f"  Points count: {N}")
    print(f"  SourceID range : [{min_s:.0f}, {max_s:.0f}]")
    print(f"  Hop1_ID  range : [{min_h1:.0f}, {max_h1:.0f}]")
    print(f"  Hop2_ID  range : [{min_h2:.0f}, {max_h2:.0f}]")
    if errors:
        print(f"  [FAIL] {errors} point mismatches", file=sys.stderr)
        return False
    print("  Points  : PASS")
    return True


def verify_offsets(offsets_path: Path, dim0: int, dim1: int, dim2: int) -> bool:
    N = dim0 * dim1 * dim2
    data = offsets_path.read_bytes()
    expected_bytes = N * 8
    if len(data) != expected_bytes:
        print(f"  [FAIL] offsets file size {len(data)} != expected {expected_bytes}", file=sys.stderr)
        return False

    view = memoryview(data).cast('Q')
    errors = 0
    for i in range(N):
        if view[i] != i:
            if errors < 5:
                print(f"  [FAIL] offset[{i}] = {view[i]}, expected {i}", file=sys.stderr)
            errors += 1

    print(f"  Offsets count  : {N}")
    print(f"  Offsets range  : [0, {N-1}]")
    if errors:
        print(f"  [FAIL] {errors} offset mismatches", file=sys.stderr)
        return False
    print("  Offsets : PASS")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dim0, dim1, dim2 = args.dims
    N = dim0 * dim1 * dim2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_pts_path = out_dir / args.points_name
    offsets_path = out_dir / args.offsets_name

    print("=" * 60)
    print("True3D 1M Dataset Generator")
    print(f"  Grid     : {dim0} x {dim1} x {dim2} = {N:,} points")
    print(f"  SourceID : [0, {dim0-1}]")
    print(f"  Hop1_ID  : [0, {dim1-1}]")
    print(f"  Hop2_ID  : [0, {dim2-1}]")
    print(f"  Offset   = SourceID*{dim1*dim2} + Hop1_ID*{dim2} + Hop2_ID")
    print(f"  Points   -> {raw_pts_path}  (raw float64, {N*3*8/1024/1024:.1f} MB)")
    print(f"  Offsets  -> {offsets_path}  (raw uint64, {N*8/1024/1024:.1f} MB)")
    print("=" * 60)

    print("\nGenerating data...")
    pts_bytes, off_bytes = generate(dim0, dim1, dim2)

    print("Writing points file...")
    raw_pts_path.write_bytes(pts_bytes)
    print(f"  -> {raw_pts_path}  ({raw_pts_path.stat().st_size/1024/1024:.1f} MB)")

    print("Writing offsets file...")
    offsets_path.write_bytes(off_bytes)
    print(f"  -> {offsets_path}  ({offsets_path.stat().st_size/1024/1024:.1f} MB)")

    # Try TPIE conversion if a helper binary is supplied
    if args.tpie_binary:
        tpie_path = out_dir / args.points_name.replace(".raw", ".tpie")
        print(f"\nConverting to TPIE via {args.tpie_binary}...")
        import subprocess
        result = subprocess.run(
            [args.tpie_binary, str(raw_pts_path), str(tpie_path), str(N), "3"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [WARN] conversion failed: {result.stderr}", file=sys.stderr)
        else:
            print(f"  -> {tpie_path}")

    if args.verify:
        print("\n--- Verification ---")
        ok_pts = verify_raw_points(raw_pts_path, dim0, dim1, dim2)
        ok_off = verify_offsets(offsets_path, dim0, dim1, dim2)
        if ok_pts and ok_off:
            print("\nAll verifications PASSED.")
        else:
            print("\nVerification FAILED.", file=sys.stderr)
            sys.exit(1)

    print("\nNext step: convert the raw points file to TPIE format.")
    print("  See docs/EXPERIMENTS.md for the exact command using raw_to_tpie.")
    print("Done.")


if __name__ == "__main__":
    main()
