#!/usr/bin/env python3
"""
Generate and validate 3D learned-index datasets for graph-path keys.

Usage examples:
  python scripts/generate_3d_datasets.py
  python scripts/generate_3d_datasets.py --out_dir out --n 10000 --seed 42 --write_bin
  python scripts/generate_3d_datasets.py --dense_dims 20 25 20 --sparse_universe 2000 2000 2000
  python scripts/generate_3d_datasets.py --validate_strict

Output layout:
  out/
    dense_3d_10k/
      dense_3d_10k.csv
      dense_3d_10k.bin   (if --write_bin)
      stats.json
    sparse_3d_10k/
      sparse_3d_10k.csv
      sparse_3d_10k.bin  (if --write_bin)
      stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Record = Tuple[int, int, int, int]


def _format_tag(n: int) -> str:
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def generate_dense_records(n: int, dims: Sequence[int]) -> List[Record]:
    n_src, n_h1, n_h2 = dims
    if n_src * n_h1 * n_h2 != n:
        raise ValueError(
            f"Dense dims product mismatch: {n_src}*{n_h1}*{n_h2}={n_src*n_h1*n_h2}, expected n={n}."
        )

    rows: List[Record] = []
    offset = 0
    for src in range(n_src):
        for h1 in range(n_h1):
            for h2 in range(n_h2):
                rows.append((src, h1, h2, offset))
                offset += 1
    return rows


def _sample_unique_triples(
    n: int,
    universe: Sequence[int],
    rng: random.Random,
) -> List[Tuple[int, int, int]]:
    u_src, u_h1, u_h2 = universe
    universe_size = u_src * u_h1 * u_h2
    if n > universe_size:
        raise ValueError(f"n={n} exceeds universe size={universe_size}.")

    seen = set()
    while len(seen) < n:
        seen.add((
            rng.randrange(u_src),
            rng.randrange(u_h1),
            rng.randrange(u_h2),
        ))
    return list(seen)


def generate_sparse_records(n: int, universe: Sequence[int], seed: int) -> List[Record]:
    rng = random.Random(seed)
    triples = _sample_unique_triples(n=n, universe=universe, rng=rng)
    triples.sort()
    rows: List[Record] = []
    for i, (src, h1, h2) in enumerate(triples):
        rows.append((src, h1, h2, i))
    return rows


def write_csv(path: Path, rows: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SourceID", "Hop1_ID", "Hop2_ID", "Offset"])
        w.writerows(rows)


def write_bin_u64x4(path: Path, rows: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pack = struct.Struct("<QQQQ")
    with path.open("wb") as f:
        for src, h1, h2, off in rows:
            f.write(pack.pack(src, h1, h2, off))


def _check_or_raise(condition: bool, message: str, strict: bool) -> bool:
    if condition:
        return True
    if strict:
        raise AssertionError(message)
    return False


def _head_tail(rows: Sequence[Record], k: int = 5) -> Tuple[List[Record], List[Record]]:
    if len(rows) <= k:
        return list(rows), list(rows)
    return list(rows[:k]), list(rows[-k:])


def _basic_stats(rows: Sequence[Record]) -> Dict[str, object]:
    srcs = [r[0] for r in rows]
    h1s = [r[1] for r in rows]
    h2s = [r[2] for r in rows]
    offs = [r[3] for r in rows]
    h, t = _head_tail(rows, k=5)
    return {
        "count": len(rows),
        "min": {
            "SourceID": min(srcs) if rows else None,
            "Hop1_ID": min(h1s) if rows else None,
            "Hop2_ID": min(h2s) if rows else None,
            "Offset": min(offs) if rows else None,
        },
        "max": {
            "SourceID": max(srcs) if rows else None,
            "Hop1_ID": max(h1s) if rows else None,
            "Hop2_ID": max(h2s) if rows else None,
            "Offset": max(offs) if rows else None,
        },
        "head": h,
        "tail": t,
    }


def validate_common(rows: Sequence[Record], n: int, strict: bool) -> Dict[str, object]:
    checks: Dict[str, bool] = {}

    checks["row_count_eq_n"] = _check_or_raise(
        len(rows) == n,
        f"Row count mismatch: got {len(rows)}, expected {n}",
        strict,
    )

    seen = set()
    duplicate_found = False
    for src, h1, h2, _ in rows:
        key = (src, h1, h2)
        if key in seen:
            duplicate_found = True
            break
        seen.add(key)

    checks["no_duplicate_keys"] = _check_or_raise(
        not duplicate_found,
        "Duplicate (SourceID,Hop1_ID,Hop2_ID) found.",
        strict,
    )

    lex_sorted = True
    for i in range(1, len(rows)):
        if rows[i - 1][:3] >= rows[i][:3]:
            lex_sorted = False
            break

    checks["strict_lexicographic_sorted"] = _check_or_raise(
        lex_sorted,
        "Rows are not strictly lexicographically sorted by (SourceID,Hop1_ID,Hop2_ID).",
        strict,
    )

    offset_ok = True
    for i, (_, _, _, off) in enumerate(rows):
        if off != i:
            offset_ok = False
            break

    checks["offset_equals_row_index"] = _check_or_raise(
        offset_ok,
        "Offset mismatch: expected Offset == row index for all rows.",
        strict,
    )

    out = {
        "common": {
            "checks": checks,
            "distinct_counts": {
                "SourceID": len({r[0] for r in rows}),
                "Hop1_ID": len({r[1] for r in rows}),
                "Hop2_ID": len({r[2] for r in rows}),
            },
        },
    }
    out.update(_basic_stats(rows))
    return out


def validate_dense(rows: Sequence[Record], n: int, dims: Sequence[int], strict: bool) -> Dict[str, object]:
    out = validate_common(rows, n=n, strict=strict)
    checks: Dict[str, bool] = {}

    s_vals = sorted({r[0] for r in rows})
    h1_vals = sorted({r[1] for r in rows})
    h2_vals = sorted({r[2] for r in rows})
    ds, dh1, dh2 = len(s_vals), len(h1_vals), len(h2_vals)

    observed_grid = ds * dh1 * dh2
    coverage_ratio = (n / observed_grid) if observed_grid else 0.0

    checks["full_grid_distinct_product_eq_n"] = _check_or_raise(
        observed_grid == n,
        f"Distinct-product mismatch: {ds}*{dh1}*{dh2}={observed_grid}, expected {n}.",
        strict,
    )
    checks["coverage_ratio_eq_1"] = _check_or_raise(
        abs(coverage_ratio - 1.0) < 1e-12,
        f"Dense coverage ratio is {coverage_ratio}, expected 1.0.",
        strict,
    )

    observed_set = {(r[0], r[1], r[2]) for r in rows}
    expected_count = 0
    all_combos_present = True
    for s in s_vals:
        for h1 in h1_vals:
            for h2 in h2_vals:
                expected_count += 1
                if (s, h1, h2) not in observed_set:
                    all_combos_present = False
                    break
            if not all_combos_present:
                break
        if not all_combos_present:
            break

    checks["every_combination_exists"] = _check_or_raise(
        all_combos_present and expected_count == n,
        "Dense combination coverage failed: missing combos or wrong expected count.",
        strict,
    )

    out["dense"] = {
        "dims_requested": {"SourceID": dims[0], "Hop1_ID": dims[1], "Hop2_ID": dims[2]},
        "distinct_counts": {"SourceID": ds, "Hop1_ID": dh1, "Hop2_ID": dh2},
        "observed_grid": observed_grid,
        "coverage_ratio": coverage_ratio,
        "checks": checks,
    }
    return out


def validate_sparse(
    rows: Sequence[Record],
    n: int,
    universe: Sequence[int],
    strict: bool,
    coverage_universe_max: float,
    observed_grid_ratio_min: float,
) -> Dict[str, object]:
    out = validate_common(rows, n=n, strict=strict)
    checks: Dict[str, bool] = {}

    u_src, u_h1, u_h2 = universe
    universe_size = u_src * u_h1 * u_h2
    coverage_universe = n / universe_size

    checks["coverage_universe_very_small"] = _check_or_raise(
        coverage_universe < coverage_universe_max,
        (
            f"Sparse universe coverage too high: {coverage_universe} >= {coverage_universe_max}. "
            f"(Universe={universe_size}, N={n})"
        ),
        strict,
    )

    ds = len({r[0] for r in rows})
    dh1 = len({r[1] for r in rows})
    dh2 = len({r[2] for r in rows})
    observed_grid = ds * dh1 * dh2
    observed_grid_ratio = observed_grid / n

    checks["not_full_grid_observed_grid_much_larger_than_n"] = _check_or_raise(
        observed_grid_ratio > observed_grid_ratio_min,
        (
            f"Observed-grid ratio too small: {observed_grid_ratio} <= {observed_grid_ratio_min}. "
            "Dataset may not be sufficiently sparse over observed dimensions."
        ),
        strict,
    )

    out["sparse"] = {
        "universe": {"SourceID": u_src, "Hop1_ID": u_h1, "Hop2_ID": u_h2},
        "universe_size": universe_size,
        "coverage_universe": coverage_universe,
        "coverage_universe_threshold": coverage_universe_max,
        "distinct_counts": {"SourceID": ds, "Hop1_ID": dh1, "Hop2_ID": dh2},
        "observed_grid": observed_grid,
        "observed_grid_ratio": observed_grid_ratio,
        "observed_grid_ratio_threshold": observed_grid_ratio_min,
        "checks": checks,
    }
    return out


def write_stats_json(path: Path, stats: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def print_summary(name: str, stats: Dict[str, object]) -> None:
    print("=" * 72)
    print(f"DATASET: {name}")
    print(f"rows: {stats['count']}")
    print(f"min: {stats['min']}")
    print(f"max: {stats['max']}")
    print("common checks:")
    for k, v in stats["common"]["checks"].items():
        print(f"  - {k}: {'PASS' if v else 'FAIL'}")

    if "dense" in stats:
        d = stats["dense"]
        print("dense checks:")
        for k, v in d["checks"].items():
            print(f"  - {k}: {'PASS' if v else 'FAIL'}")
        print(f"dense distinct counts: {d['distinct_counts']}")
        print(f"dense observed_grid: {d['observed_grid']}")
        print(f"dense coverage_ratio: {d['coverage_ratio']:.12f}")

    if "sparse" in stats:
        s = stats["sparse"]
        print("sparse checks:")
        for k, v in s["checks"].items():
            print(f"  - {k}: {'PASS' if v else 'FAIL'}")
        print(f"sparse universe: {s['universe']} (size={s['universe_size']})")
        print(f"sparse coverage_universe: {s['coverage_universe']:.12e}")
        print(f"sparse observed_grid: {s['observed_grid']}")
        print(f"sparse observed_grid_ratio: {s['observed_grid_ratio']:.6f}")

    print("head rows:")
    for row in stats["head"]:
        print(f"  {row}")
    print("tail rows:")
    for row in stats["tail"]:
        print(f"  {row}")


def generate_and_validate(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    out_root = Path(args.out_dir)
    tag = _format_tag(args.n)

    dense_dir = out_root / f"dense_3d_{tag}"
    sparse_dir = out_root / f"sparse_3d_{tag}"

    dense_rows = generate_dense_records(args.n, args.dense_dims)
    sparse_rows = generate_sparse_records(args.n, args.sparse_universe, args.seed)

    dense_csv = dense_dir / f"dense_3d_{tag}.csv"
    sparse_csv = sparse_dir / f"sparse_3d_{tag}.csv"

    write_csv(dense_csv, dense_rows)
    write_csv(sparse_csv, sparse_rows)

    if args.write_bin:
        dense_bin = dense_dir / f"dense_3d_{tag}.bin"
        sparse_bin = sparse_dir / f"sparse_3d_{tag}.bin"
        write_bin_u64x4(dense_bin, dense_rows)
        write_bin_u64x4(sparse_bin, sparse_rows)

    dense_stats = validate_dense(
        dense_rows,
        n=args.n,
        dims=args.dense_dims,
        strict=args.validate_strict,
    )
    sparse_stats = validate_sparse(
        sparse_rows,
        n=args.n,
        universe=args.sparse_universe,
        strict=args.validate_strict,
        coverage_universe_max=args.sparse_coverage_max,
        observed_grid_ratio_min=args.sparse_observed_grid_ratio_min,
    )

    write_stats_json(dense_dir / "stats.json", dense_stats)
    write_stats_json(sparse_dir / "stats.json", sparse_stats)

    print_summary(f"dense_3d_{tag}", dense_stats)
    print_summary(f"sparse_3d_{tag}", sparse_stats)

    return {"dense": dense_stats, "sparse": sparse_stats}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and validate dense/sparse 3D learned-index datasets.")
    p.add_argument("--out_dir", default="out", help="Output directory root.")
    p.add_argument("--n", type=int, default=10000, help="Number of rows per dataset.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sparse dataset generation.")
    p.add_argument(
        "--dense_dims",
        nargs=3,
        type=int,
        metavar=("N_SRC", "N_HOP1", "N_HOP2"),
        default=[20, 25, 20],
        help="Dense grid dimensions (must multiply to n).",
    )
    p.add_argument(
        "--sparse_universe",
        nargs=3,
        type=int,
        metavar=("U_SRC", "U_HOP1", "U_HOP2"),
        default=[2000, 2000, 2000],
        help="Sparse universe sizes for sampling unique triples.",
    )
    p.add_argument("--write_bin", action="store_true", help="Also write .bin files with <QQQQ rows.")
    p.add_argument(
        "--validate_strict",
        action="store_true",
        help="Raise assertion errors for failed validations. Without this, stats still report PASS/FAIL.",
    )
    p.add_argument(
        "--sparse_coverage_max",
        type=float,
        default=1e-6,
        help="Validation threshold: sparse coverage_universe must be < this value.",
    )
    p.add_argument(
        "--sparse_observed_grid_ratio_min",
        type=float,
        default=5.0,
        help="Validation threshold: observed_grid / N must be > this value.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.n <= 0:
        raise ValueError("--n must be > 0")
    if any(v <= 0 for v in args.dense_dims):
        raise ValueError("--dense_dims values must be > 0")
    if any(v <= 0 for v in args.sparse_universe):
        raise ValueError("--sparse_universe values must be > 0")

    generate_and_validate(args)


if __name__ == "__main__":
    main()
