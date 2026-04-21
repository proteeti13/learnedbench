#!/usr/bin/env python3
"""
combine_zmindex_results.py
Merge all per-run CSVs from build/results/zmindex_per_run/ into one combined CSV
and print a human-readable summary table.

Run from repo root:
    python3 scripts/combine_zmindex_results.py
"""

import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PER_RUN_DIR = REPO / "build" / "results" / "zmindex_per_run"
OUT_CSV     = REPO / "zmindex_all_datasets_combined.csv"

COLUMNS = [
    "dataset_name", "data_source", "distribution", "graph_type",
    "full_dataset_size", "subset_size",
    "build_time_s", "index_size_mb",
    "mean_latency_us", "p95_latency_us", "throughput_qps",
    "avg_pgm_refine_window", "correctness_pct",
]

# Canonical dataset order to match RSMI CSV
DATASET_ORDER = [
    "wiki_vote", "roadnet_ca", "web_google",
    "uniform_sparse", "uniform_dense", "uniform_matched",
    "normal", "lognormal",
]

def main():
    if not PER_RUN_DIR.exists():
        print(f"ERROR: {PER_RUN_DIR} does not exist. Run run_zmindex_all.sh first.")
        sys.exit(1)

    rows = []
    missing = []

    csv_files = sorted(PER_RUN_DIR.glob("zm_*.csv"))
    if not csv_files:
        print(f"ERROR: no CSV files found in {PER_RUN_DIR}")
        sys.exit(1)

    for fpath in csv_files:
        with open(fpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    if not rows:
        print("ERROR: all CSV files were empty.")
        sys.exit(1)

    # Sort by dataset order then subset_size
    def sort_key(r):
        ds = r["dataset_name"]
        idx = DATASET_ORDER.index(ds) if ds in DATASET_ORDER else 99
        return (idx, int(r["subset_size"]))

    rows.sort(key=sort_key)

    # Write combined CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})

    print(f"Combined CSV written → {OUT_CSV}  ({len(rows)} rows)\n")

    # Print summary table
    header = (
        f"{'Dataset':<20} {'N':>12} {'Build(s)':>10} {'IdxMB':>8} "
        f"{'Mean(µs)':>10} {'P95(µs)':>9} {'QPS':>10} "
        f"{'PGMwin':>8} {'Corr%':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    prev_ds = None
    for r in rows:
        ds = r["dataset_name"]
        if prev_ds and ds != prev_ds:
            print()
        prev_ds = ds
        n   = int(r["subset_size"])
        n_s = f"{n:,}"
        print(
            f"{ds:<20} {n_s:>12} "
            f"{float(r['build_time_s']):>10.3f} "
            f"{float(r['index_size_mb']):>8.3f} "
            f"{float(r['mean_latency_us']):>10.4f} "
            f"{float(r['p95_latency_us']):>9.4f} "
            f"{int(float(r['throughput_qps'])):>10,} "
            f"{float(r['avg_pgm_refine_window']):>8.2f} "
            f"{float(r['correctness_pct']):>7.2f}"
        )

    print(sep)

    # Warn about missing
    expected_labels = {
        "wiki_vote": [1000000, 2500000, 4542805],
        "roadnet_ca": [1000000, 2500000, 17523394],
        "web_google": [1000000, 2500000, 60687836],
        "uniform_sparse": [1000000, 5000000, 10000000],
        "uniform_dense": [1000000, 5000000, 10000000],
        "uniform_matched": [1000000, 5000000, 10000000],
        "normal": [1000000, 5000000, 10000000],
        "lognormal": [1000000, 5000000, 10000000],
    }
    found = {(r["dataset_name"], int(r["subset_size"])) for r in rows}
    for ds, sizes in expected_labels.items():
        for sz in sizes:
            if (ds, sz) not in found:
                missing.append(f"  MISSING: {ds} N={sz:,}")

    if missing:
        print("\nWARNING — the following runs were not found:")
        for m in missing:
            print(m)
    else:
        print(f"\nAll 24 expected runs present. ✓")

if __name__ == "__main__":
    main()
