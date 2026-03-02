"""
generate_dense_3d.py  –  LexFlattenBaseline Dataset Generator
=============================================================
Generates a dense 3D grid of (SourceID, Hop1_ID, Hop2_ID) tuples in
lexicographic order (SourceID major, Hop1 middle, Hop2 minor), assigns
each tuple a sequential Offset, and writes a semicolon-delimited CSV.

Output: data/dense_3d_with_offset.csv  (relative to this script's directory)
Format: SourceID;Hop1_ID;Hop2_ID;Offset  (header line included)

Default grid: 10 × 10 × 100 = 10,000 rows  (matches the 10K benchmark dataset)

Usage:
    python3 generate_dense_3d.py
    # or from repo root:
    python3 tools/datasets/lex_flatten_baseline/generate_dense_3d.py
"""

import csv
from pathlib import Path


def generate_dense_3d(
    out_dir="data",
    n_src=10,
    n_hop1=10,
    n_hop2=100,
    write_csv=True,
    csv_filename="dense_3d_with_offset.csv",
    csv_delimiter=";",
):
    """
    Generates a dense 3D grid of (SourceID, Hop1_ID, Hop2_ID) with lexicographic order:
        SourceID major, Hop1 middle, Hop2 minor.
    Offset is the sequential rank in this order (0-indexed).

    Total tuples = n_src * n_hop1 * n_hop2.
    Default: 10 * 10 * 100 = 10,000 rows.
    """

    total = n_src * n_hop1 * n_hop2

    # Resolve output directory relative to this script's location
    script_dir = Path(__file__).parent
    out_path = script_dir / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / csv_filename

    csv_file = None
    csv_writer = None
    if write_csv:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file, delimiter=csv_delimiter)
        csv_writer.writerow(["SourceID", "Hop1_ID", "Hop2_ID", "Offset"])

    offset = 0

    try:
        # Lexicographic order by construction: SourceID → Hop1 → Hop2
        for src in range(n_src):
            for hop1 in range(n_hop1):
                for hop2 in range(n_hop2):
                    if write_csv:
                        csv_writer.writerow([src, hop1, hop2, offset])
                    offset += 1

        print(f"[OK] Generated {offset:,} tuples  ({n_src}×{n_hop1}×{n_hop2} grid).")
        if write_csv:
            print(f"[OK] CSV written: {csv_path}")

    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    # Default: 10 × 10 × 100 = 10,000 rows (matches the LexFlattenBaseline 10K experiment)
    generate_dense_3d(
        out_dir="data",
        n_src=10,
        n_hop1=10,
        n_hop2=100,
        write_csv=True,
        csv_delimiter=";",
    )
