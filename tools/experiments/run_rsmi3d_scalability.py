#!/usr/bin/env python3
"""
tools/experiments/run_rsmi3d_scalability.py
--------------------------------------------
Run RSMI3D scalability benchmarks across N = 10K, 100K, 1M for dense and
sparse distributions. Parses benchmark stdout, writes a CSV, and prints a
summary table to the console.

Usage:
  cd /path/to/learnedbench
  python3 tools/experiments/run_rsmi3d_scalability.py [--out_csv build/results/rsmi3d_scalability.csv]

Optional:
  --epsilon  INT   PGM epsilon (default 8)
  --queries  INT   queries per run (default 100000)
  --seed     INT   RNG seed (default 42)
  --fanout   INT   RSMI3D fanout (default 10)
  --depth    INT   max_depth (default 1)
  --min_leaf INT   min points per leaf (default 100)
  --model    STR   pgm|linear|midpoint (default pgm)
  --dry_run        print commands only, do not execute
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH     = REPO_ROOT / "build/bin/bench_rsmi3d_true3d"
DATA      = REPO_ROOT / "build/data"
RESULTS   = REPO_ROOT / "build/results"

# ---------------------------------------------------------------------------
# Experiment table
# Each entry: (dataset_name, N, distribution, pts_file, offs_file)
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    ("Dense3D_10K",    10_000,   "dense",
     "dense3d_10k_points.tpie",   "dense3d_10k_offsets.bin"),
    ("Sparse3D_10K",   10_000,   "sparse",
     "sparse3d_10k_points.tpie",  "sparse3d_10k_offsets.bin"),
    ("Dense3D_100K",  100_000,   "dense",
     "dense3d_100k_points.tpie",  "dense3d_100k_offsets.bin"),
    ("Sparse3D_100K", 100_000,   "sparse",
     "sparse3d_100k_points.tpie", "sparse3d_100k_offsets.bin"),
    ("Dense3D_1M",  1_000_000,   "dense",
     "dense3d_1m_points.tpie",    "dense3d_1m_offsets.bin"),
    ("Sparse3D_1M", 1_000_000,   "sparse",
     "sparse3d_1m_points.tpie",   "sparse3d_1m_offsets.bin"),
]

# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def _get(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else default


def parse_output(output: str) -> dict[str, str]:
    """
    Extract all metrics from bench_rsmi3d_true3d stdout.
    Returns a flat string-valued dict.
    """
    # Build time: "  Build time          : 0.101 s  (101 ms)"
    build_ms = _get(r"Build time\s*:\s*[\d.]+\s*s\s*\(([\d.]+)\s*ms\)", output)

    # Index size: "  Index size          : 16401472 bytes  (...)"
    idx_bytes = _get(r"Index size\s*:\s*(\d+)\s*bytes", output)

    # Latencies
    lat_mean = _get(r"Latency\s+mean\s*:\s*(\d+)\s*ns",  output)
    lat_p50  = _get(r"Latency\s+p50\s*:\s*(\d+)\s*ns",   output)
    lat_p95  = _get(r"Latency\s+p95\s*:\s*(\d+)\s*ns",   output)
    lat_p99  = _get(r"Latency\s+p99\s*:\s*(\d+)\s*ns",   output)

    # Throughput: "    Throughput          : 3248123 queries/sec"
    throughput = _get(r"Throughput\s*:\s*(\d+)\s*queries/sec", output)

    # PredError block (titled lines)
    pe_mean = _get(r"PredError mean\s*:\s*([\d.]+)",  output)
    pe_p95  = _get(r"PredError p95\s*:\s*(\d+)",      output)
    pe_p99  = _get(r"PredError p99\s*:\s*(\d+)",      output)
    pe_max  = _get(r"PredError max\s*:\s*(\d+)",      output)

    # RefineWindow block
    rw_mean = _get(r"RefineWindow mean\s*:\s*([\d.]+)", output)
    rw_p95  = _get(r"RefineWindow p95\s*:\s*(\d+)",    output)
    rw_p99  = _get(r"RefineWindow p99\s*:\s*(\d+)",    output)

    # ZMI-compatible summary line
    cs_avg = _get(r"CorrSteps\(avg\)=([\d.]+)",   output)
    cs_max = _get(r"CorrSteps\(max\)=(\d+)",       output)

    # Correctness: "    Correctness         : 100000 / 100000  (100.00%)"
    corr_pct = _get(r"Correctness\s*:\s*\d+\s*/\s*\d+\s*\(([\d.]+)%\)", output)

    return {
        "build_time_ms":    build_ms,
        "index_size_bytes": idx_bytes,
        "latency_mean_ns":  lat_mean,
        "latency_p50_ns":   lat_p50,
        "latency_p95_ns":   lat_p95,
        "latency_p99_ns":   lat_p99,
        "throughput_qps":   throughput,
        "prederror_mean":   pe_mean,
        "prederror_p95":    pe_p95,
        "prederror_p99":    pe_p99,
        "prederror_max":    pe_max,
        "refinewindow_mean": rw_mean,
        "refinewindow_p95": rw_p95,
        "refinewindow_p99": rw_p99,
        "corrsteps_avg":    cs_avg,
        "corrsteps_max":    cs_max,
        "correctness_pct":  corr_pct,
    }


# ---------------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------------

def run_one(exp: tuple, params: dict, dry_run: bool) -> dict | None:
    name, n, dist, pts_f, offs_f = exp
    pts_path  = DATA / pts_f
    offs_path = DATA / offs_f

    if not pts_path.exists():
        print(f"  [SKIP] {name}: {pts_path} not found", file=sys.stderr)
        return None
    if not offs_path.exists():
        print(f"  [SKIP] {name}: {offs_path} not found", file=sys.stderr)
        return None

    cmd = [
        str(BENCH),
        str(pts_path),
        str(offs_path),
        str(n),
        name,
        "--epsilon",  str(params["epsilon"]),
        "--fanout",   str(params["fanout"]),
        "--max_depth",str(params["max_depth"]),
        "--min_leaf", str(params["min_leaf"]),
        "--model",    params["model"],
        "--queries",  str(params["queries"]),
        "--seed",     str(params["seed"]),
    ]

    print(f"\n{'='*60}")
    print(f"  {name}  (N={n:,}, dist={dist})")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(f"  [ERROR] {name} exited {result.returncode}:\n{result.stderr}",
              file=sys.stderr)
        return None

    metrics = parse_output(result.stdout)
    metrics.update({
        "dataset_name":  name,
        "N":             str(n),
        "distribution":  dist,
        "epsilon":       str(params["epsilon"]),
        "fanout":        str(params["fanout"]),
        "max_depth":     str(params["max_depth"]),
        "min_leaf":      str(params["min_leaf"]),
        "model":         params["model"],
        "queries":       str(params["queries"]),
    })
    return metrics


# ---------------------------------------------------------------------------
# CSV fieldname order
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "dataset_name", "N", "distribution",
    "epsilon", "fanout", "max_depth", "min_leaf", "model", "queries",
    "build_time_ms", "index_size_bytes",
    "latency_mean_ns", "latency_p50_ns", "latency_p95_ns", "latency_p99_ns",
    "throughput_qps",
    "prederror_mean", "prederror_p95", "prederror_p99", "prederror_max",
    "refinewindow_mean", "refinewindow_p95", "refinewindow_p99",
    "corrsteps_avg", "corrsteps_max",
    "correctness_pct",
]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    hdr = (f"{'Dataset':<20} {'N':>10} {'Dist':<7} "
           f"{'Build(ms)':>9} {'IdxMB':>6} "
           f"{'LatMean':>8} {'LatP99':>7} "
           f"{'Thpt(K)':>8} "
           f"{'PE_max':>7} {'RW_mean':>8} {'Corr%':>6}")
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for r in rows:
        idx_mb  = float(r.get("index_size_bytes", 0)) / 1024 / 1024
        thpt_k  = int(r.get("throughput_qps", 0)) // 1_000
        print(
            f"{r['dataset_name']:<20} {int(r['N']):>10,} {r['distribution']:<7} "
            f"{r['build_time_ms']:>9} {idx_mb:>6.1f} "
            f"{r['latency_mean_ns']:>8} {r['latency_p99_ns']:>7} "
            f"{thpt_k:>8} "
            f"{r['prederror_max']:>7} {r['refinewindow_mean']:>8} "
            f"{r['correctness_pct']:>6}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RSMI3D scalability benchmark suite.")
    parser.add_argument("--out_csv",  default=str(RESULTS / "rsmi3d_scalability.csv"))
    parser.add_argument("--epsilon",  type=int, default=8)
    parser.add_argument("--queries",  type=int, default=100_000)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--fanout",   type=int, default=10)
    parser.add_argument("--depth",    type=int, default=1,
                        dest="max_depth")
    parser.add_argument("--min_leaf", type=int, default=100)
    parser.add_argument("--model",    default="pgm")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    params = {
        "epsilon":   args.epsilon,
        "queries":   args.queries,
        "seed":      args.seed,
        "fanout":    args.fanout,
        "max_depth": args.max_depth,
        "min_leaf":  args.min_leaf,
        "model":     args.model,
    }

    print(f"\nRSMI3D Scalability Benchmark Suite")
    print(f"  epsilon={params['epsilon']}  fanout={params['fanout']}  "
          f"max_depth={params['max_depth']}  min_leaf={params['min_leaf']}  "
          f"model={params['model']}  Q={params['queries']:,}")
    print(f"  Experiments: {len(EXPERIMENTS)}")

    rows: list[dict] = []
    for exp in EXPERIMENTS:
        row = run_one(exp, params, dry_run=args.dry_run)
        if row:
            rows.append(row)

    if args.dry_run:
        print("\n[dry_run] No results collected.")
        return 0

    if not rows:
        print("No results collected — check that datasets exist.", file=sys.stderr)
        return 1

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written: {out_csv}")
    print_summary(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
