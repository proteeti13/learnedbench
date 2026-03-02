#!/usr/bin/env python3
"""
Compatibility checker for graph 3D CSV datasets vs learnedbench xmindex/zmindex pipeline.

This tool inspects the local repo to discover the true input format expected by the
benchmark/data loader, then validates a CSV and reports compatibility.

Key repo format references used by this checker:
- bench/bench.cpp: benchmark entrypoint and index registry ("zm", etc.)
- utils/datautils.hpp: read_points(...) loader using tpie::file_stream<double>
- indexes/learned/zmindex.hpp: ZMIndex uses point tuples; no payload/value column
- CMakeLists.txt: BENCH_DIM targets (e.g., bench2d_default, bench3d_toronto)
- utils/datagen.cpp: CSV->TPIE conversion utility (process_csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


CANONICAL_COLS = ("SourceID", "Hop1_ID", "Hop2_ID", "Offset")

ALIASES = {
    "SourceID": {"sourceid", "source_id", "src", "source"},
    "Hop1_ID": {"hop1_id", "hop1", "h1", "dest", "dst"},
    "Hop2_ID": {"hop2_id", "hop2", "h2"},
    "Offset": {"offset", "rank", "position", "global_offset", "idx", "index"},
}


@dataclass
class CsvValidation:
    path: str
    row_count: int
    header_ok: bool
    header_mapping: Dict[str, str]
    sorted_lex: bool
    no_duplicates: bool
    offset_matches_row_index: bool
    min_vals: Dict[str, Optional[int]]
    max_vals: Dict[str, Optional[int]]
    head_rows: List[Tuple[int, int, int, int]]
    tail_rows: List[Tuple[int, int, int, int]]
    errors: List[str]

    @property
    def passed(self) -> bool:
        return (
            self.header_ok
            and self.sorted_lex
            and self.no_duplicates
            and self.offset_matches_row_index
            and not self.errors
        )


@dataclass
class RepoExpectation:
    index_kind_input: str
    index_kind_normalized: str
    index_available: bool
    xmindex_symbol_found: bool
    accepted_formats: List[str]
    direct_csv_supported: bool
    loader_function: str
    loader_file: str
    loader_lines: List[int]
    record_layout: str
    payload_expected: str
    sorting_required_by_loader: str
    compile_time_dim_note: str
    proof: List[str]


@dataclass
class CompatibilityVerdict:
    directly_compatible: bool
    verdict: str
    reason: str
    conversion_required: Optional[Dict[str, object]]


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def run_rg(repo_root: Path, pattern: str, paths: Sequence[str]) -> List[str]:
    cmd = ["rg", "-n", pattern, *paths]
    try:
        out = subprocess.run(cmd, cwd=repo_root, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return []
    if out.returncode not in (0, 1):
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def discover_repo_expectations(repo_root: Path, index_kind: str, verbose: bool = False) -> RepoExpectation:
    kind_in = index_kind.strip()
    kind_norm = kind_in.lower().replace("-", "").replace("_", "")

    # Canonicalize known spelling for this repo
    if kind_norm in {"zmindex", "zm"}:
        canonical_idx = "zm"
    elif kind_norm in {"xmindex", "xm"}:
        canonical_idx = "xm"
    else:
        canonical_idx = kind_norm

    bench_cpp = repo_root / "bench/bench.cpp"
    datautils_hpp = repo_root / "utils/datautils.hpp"
    zmindex_hpp = repo_root / "indexes/learned/zmindex.hpp"
    cmake_txt = repo_root / "CMakeLists.txt"
    datagen_cpp = repo_root / "utils/datagen.cpp"

    proof: List[str] = []
    loader_lines = []

    # Check registry availability from bench.cpp
    reg_hits = run_rg(repo_root, r'idx_name\.compare\("([^"]+)"\)', ["bench/bench.cpp"])
    available_idx = set()
    for line in reg_hits:
        m = re.search(r'idx_name\.compare\("([^"]+)"\)', line)
        if m:
            available_idx.add(m.group(1))

    index_available = canonical_idx in available_idx

    xm_hits = run_rg(
        repo_root,
        r"\bxmindex\b|\bxm-index\b|\bXMIndex\b|\bXM-Index\b",
        ["bench", "indexes", "scripts", "README.md", "CMakeLists.txt"],
    )
    xmindex_symbol_found = len(xm_hits) > 0

    # Loader proof
    rp_hits = run_rg(repo_root, r"read_points\(points, fname, N\)", ["bench/bench.cpp"])
    if rp_hits:
        proof.extend(rp_hits)

    tpie_hits = run_rg(repo_root, r"tpie::file_stream<double>|read_points\(|#include <tpie/", ["utils/datautils.hpp"])
    if tpie_hits:
        proof.extend(tpie_hits)

    loader_line_hits = run_rg(repo_root, r"template<size_t dim>\s*|inline void read_points", ["utils/datautils.hpp"])
    for h in loader_line_hits:
        # rg output may be either "<line>:<text>" (single-file search) or "path:<line>:<text>"
        m_line_only = re.match(r"^(\d+):", h)
        m_path_line = re.match(r"^[^:]+:(\d+):", h)
        if m_line_only:
            loader_lines.append(int(m_line_only.group(1)))
        elif m_path_line:
            loader_lines.append(int(m_path_line.group(1)))

    zm_hits = run_rg(repo_root, r"using ZM =|ZMIndex<|tuple|payload|size_t", ["indexes/learned/zmindex.hpp"])
    if zm_hits:
        proof.extend(zm_hits)

    dim_hits = run_rg(repo_root, r"bench2d_default|bench3d_toronto|BENCH_DIM", ["CMakeLists.txt", "bench/bench.cpp"])
    if dim_hits:
        proof.extend(dim_hits)

    # datagen CSV converter support proof
    datagen_hits = run_rg(repo_root, r"process_csv|csv_to_bin", ["utils/datagen.cpp", "utils/datautils.hpp"])
    if datagen_hits:
        proof.extend(datagen_hits)

    # Deduplicate proof lines, keep stable order
    seen = set()
    proof = [p for p in proof if not (p in seen or seen.add(p))]

    accepted_formats = [
        "TPIE file_stream<double> dataset file (custom container; not plain CSV)",
        "Optional input CSV only via pre-conversion utility (utils/datagen.cpp -> csv_to_bin)",
    ]

    record_layout = (
        "Per point: Dim values of type double (8 bytes each), read sequentially as [x0, x1, ..., x{Dim-1}] "
        "by read_points(...). For 3D benchmark target, Dim=3 => 3x float64 per point."
    )

    payload_expected = (
        "No payload/value column consumed by loader. ZM-Index builds from point coordinates only; "
        "Offset is not read by bench loader."
    )

    sorting_required_by_loader = (
        "Loader does not enforce sortedness; it reads points in file order. "
        "Any sort requirement is external/semantic, not mandated by read_points()."
    )

    compile_time_dim_note = (
        "Dimensionality is compile-time in benchmark binaries (e.g., bench2d_default uses BENCH_DIM=2, "
        "bench3d_toronto uses BENCH_DIM=3). Dataset must match the chosen binary's Dim."
    )

    if verbose:
        print(f"[debug] discovered indices in bench registry: {sorted(available_idx)}")

    return RepoExpectation(
        index_kind_input=kind_in,
        index_kind_normalized=canonical_idx,
        index_available=index_available,
        xmindex_symbol_found=xmindex_symbol_found,
        accepted_formats=accepted_formats,
        direct_csv_supported=False,
        loader_function="bench::utils::read_points (tpie::file_stream<double>)",
        loader_file="utils/datautils.hpp",
        loader_lines=sorted(set(loader_lines)),
        record_layout=record_layout,
        payload_expected=payload_expected,
        sorting_required_by_loader=sorting_required_by_loader,
        compile_time_dim_note=compile_time_dim_note,
        proof=proof,
    )


def map_header(fieldnames: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    errors: List[str] = []
    normalized_to_original = {normalize_col(f): f for f in fieldnames}
    mapping: Dict[str, str] = {}

    for c in CANONICAL_COLS:
        candidates = {normalize_col(c), *{normalize_col(a) for a in ALIASES[c]}}
        found = None
        for cand in candidates:
            if cand in normalized_to_original:
                found = normalized_to_original[cand]
                break
        if found is None:
            errors.append(f"Missing required column for {c}. Header fields={list(fieldnames)}")
        else:
            mapping[c] = found

    return mapping, errors


def parse_int_strict(value: str, field: str, row_num: int) -> int:
    s = value.strip()
    if s == "":
        raise ValueError(f"row {row_num}: empty value for {field}")
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    # allow float string only if it is an exact integer like 12.0
    try:
        f = float(s)
    except ValueError as ex:
        raise ValueError(f"row {row_num}: non-numeric value for {field}: {value!r}") from ex
    if not f.is_integer():
        raise ValueError(f"row {row_num}: non-integer value for {field}: {value!r}")
    return int(f)


def validate_csv(csv_path: Path, dry_run: bool = False) -> CsvValidation:
    errors: List[str] = []
    row_count = 0
    header_ok = False
    header_mapping: Dict[str, str] = {}

    sorted_lex = True
    no_duplicates = True
    offset_matches_row_index = True

    mins = {c: None for c in CANONICAL_COLS}
    maxs = {c: None for c in CANONICAL_COLS}

    head_rows: List[Tuple[int, int, int, int]] = []
    tail_rows: List[Tuple[int, int, int, int]] = []

    seen = set()
    prev_key: Optional[Tuple[int, int, int]] = None

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            errors.append("CSV appears empty or missing header row.")
        else:
            header_mapping, header_errors = map_header(reader.fieldnames)
            header_ok = len(header_errors) == 0
            errors.extend(header_errors)

        if not header_ok:
            return CsvValidation(
                path=str(csv_path),
                row_count=0,
                header_ok=False,
                header_mapping=header_mapping,
                sorted_lex=False,
                no_duplicates=False,
                offset_matches_row_index=False,
                min_vals=mins,
                max_vals=maxs,
                head_rows=[],
                tail_rows=[],
                errors=errors,
            )

        for i, row in enumerate(reader):
            row_num = i + 2  # include header line
            try:
                src = parse_int_strict(row[header_mapping["SourceID"]], "SourceID", row_num)
                h1 = parse_int_strict(row[header_mapping["Hop1_ID"]], "Hop1_ID", row_num)
                h2 = parse_int_strict(row[header_mapping["Hop2_ID"]], "Hop2_ID", row_num)
                off = parse_int_strict(row[header_mapping["Offset"]], "Offset", row_num)
            except ValueError as ex:
                errors.append(str(ex))
                if dry_run:
                    break
                continue

            rec = (src, h1, h2, off)
            row_count += 1

            if len(head_rows) < 5:
                head_rows.append(rec)
            tail_rows.append(rec)
            if len(tail_rows) > 5:
                tail_rows.pop(0)

            for cname, value in zip(CANONICAL_COLS, rec):
                if mins[cname] is None or value < mins[cname]:
                    mins[cname] = value
                if maxs[cname] is None or value > maxs[cname]:
                    maxs[cname] = value

            key = (src, h1, h2)
            if not dry_run:
                if key in seen:
                    no_duplicates = False
                seen.add(key)

                if prev_key is not None and prev_key >= key:
                    sorted_lex = False
                prev_key = key

                if off != (row_count - 1):
                    offset_matches_row_index = False

    if dry_run:
        no_duplicates = True
        sorted_lex = True
        offset_matches_row_index = True

    return CsvValidation(
        path=str(csv_path),
        row_count=row_count,
        header_ok=header_ok,
        header_mapping=header_mapping,
        sorted_lex=sorted_lex,
        no_duplicates=no_duplicates,
        offset_matches_row_index=offset_matches_row_index,
        min_vals=mins,
        max_vals=maxs,
        head_rows=head_rows,
        tail_rows=tail_rows,
        errors=errors,
    )


def build_verdict(csvv: CsvValidation, repo: RepoExpectation) -> CompatibilityVerdict:
    if not csvv.passed:
        return CompatibilityVerdict(
            directly_compatible=False,
            verdict="FAIL",
            reason="CSV validation failed.",
            conversion_required=None,
        )

    if repo.index_kind_normalized == "xm" and not repo.xmindex_symbol_found:
        return CompatibilityVerdict(
            directly_compatible=False,
            verdict="FAIL",
            reason=(
                "Requested index_kind=xmindex, but no xmindex symbol/runner was found in this repo scan. "
                "Cannot establish compatibility for xmindex in current checkout."
            ),
            conversion_required=None,
        )

    if not repo.index_available and repo.index_kind_normalized == "zm":
        return CompatibilityVerdict(
            directly_compatible=False,
            verdict="FAIL",
            reason="zm index alias not found in bench registry for this checkout.",
            conversion_required=None,
        )

    # CSV direct ingestion is not supported by bench loader.
    conversion_required = {
        "target_format": "TPIE file_stream<double> dataset file",
        "accepted_extensions": ["(no strict extension)", ".bin (common naming)"],
        "record_packing": "For 3D run: write 3 values per row as float64/double in key order [SourceID, Hop1_ID, Hop2_ID].",
        "endianness": "Managed by TPIE file_stream<double> writer; do not emit plain raw binary without TPIE container.",
        "payload_required": False,
        "payload_note": "Offset must be omitted for zmindex/bench loader input; it is not consumed.",
        "sorting_required": "Not required by loader, but current CSV already sorted lexicographically.",
        "minimal_recipe": [
            "Create a temporary CSV containing only the 3 key columns in this exact order: SourceID,Hop1_ID,Hop2_ID (no Offset).",
            "Convert to benchmark format with existing utility: build/bin/datagen --task process_csv --infname <keys3.csv> --fname <dataset_3d_tpie>",
            "Run 3D benchmark binary with zm index: build/bin/bench3d_toronto zm <dataset_3d_tpie> <N> <mode>",
        ],
    }

    return CompatibilityVerdict(
        directly_compatible=False,
        verdict="FAIL",
        reason="CSV cannot be consumed directly by zmindex benchmark loader; conversion to TPIE double-stream format is required.",
        conversion_required=conversion_required,
    )


def print_report(csvv: CsvValidation, repo: RepoExpectation, verdict: CompatibilityVerdict, verbose: bool = False) -> None:
    print("=" * 88)
    print("GRAPH3D DATASET COMPATIBILITY REPORT")
    print("=" * 88)

    print("\n[CSV validation]")
    print(f"path: {csvv.path}")
    print(f"row_count: {csvv.row_count}")
    print(f"header_mapping: {csvv.header_mapping}")
    print(f"checks:")
    print(f"  - header_ok: {'PASS' if csvv.header_ok else 'FAIL'}")
    print(f"  - lexicographic_sorted(SourceID,Hop1_ID,Hop2_ID): {'PASS' if csvv.sorted_lex else 'FAIL'}")
    print(f"  - no_duplicate_keys: {'PASS' if csvv.no_duplicates else 'FAIL'}")
    print(f"  - Offset_eq_row_index: {'PASS' if csvv.offset_matches_row_index else 'FAIL'}")
    print(f"min_vals: {csvv.min_vals}")
    print(f"max_vals: {csvv.max_vals}")
    print(f"head_rows: {csvv.head_rows}")
    print(f"tail_rows: {csvv.tail_rows}")
    if csvv.errors:
        print("errors:")
        for e in csvv.errors:
            print(f"  - {e}")

    print("\n[Repo expected format]")
    print(f"index_kind input: {repo.index_kind_input}")
    print(f"index_kind normalized: {repo.index_kind_normalized}")
    print(f"index_available_in_bench_registry: {repo.index_available}")
    print(f"xmindex_symbol_found_anywhere: {repo.xmindex_symbol_found}")
    print("accepted_formats:")
    for f in repo.accepted_formats:
        print(f"  - {f}")
    print(f"direct_csv_supported: {repo.direct_csv_supported}")
    print(f"loader_function: {repo.loader_function}")
    print(f"loader_file: {repo.loader_file}")
    print(f"loader_lines: {repo.loader_lines}")
    print(f"record_layout: {repo.record_layout}")
    print(f"payload_expected: {repo.payload_expected}")
    print(f"sorting_required_by_loader: {repo.sorting_required_by_loader}")
    print(f"compile_time_dim_note: {repo.compile_time_dim_note}")

    print("proof(file:line):")
    for p in repo.proof[:80]:
        print(f"  - {p}")
    if len(repo.proof) > 80:
        print(f"  - ... ({len(repo.proof) - 80} more lines; rerun --verbose for full context)")

    print("\n[Compatibility verdict]")
    print(f"status: {verdict.verdict}")
    print(f"directly_compatible: {verdict.directly_compatible}")
    print(f"reason: {verdict.reason}")

    if verdict.directly_compatible:
        print("next_step: DIRECTLY COMPATIBLE")
        print("run_command_template: build/bin/<bench_binary> <index_kind> <dataset_file> <N> <mode>")
    else:
        print("next_step: NOT DIRECTLY COMPATIBLE")
        if verdict.conversion_required is not None:
            print("conversion_requirements:")
            print(json.dumps(verdict.conversion_required, indent=2))

    if verbose:
        print("\n[Debug dump]")
        print(json.dumps({
            "csv_validation": asdict(csvv),
            "repo_expectation": asdict(repo),
            "verdict": asdict(verdict),
        }, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check graph-3D CSV compatibility with xmindex/zmindex benchmark pipeline.")
    p.add_argument("--csv", required=True, help="Path to CSV dataset with SourceID,Hop1_ID,Hop2_ID,Offset header.")
    p.add_argument("--index_kind", default="xmindex", help="Index kind to check, e.g. xmindex or zmindex.")
    p.add_argument("--verbose", action="store_true", help="Print extra debug details.")
    p.add_argument("--dry_run", action="store_true", help="Skip expensive full-row checks (duplicates/sorted/offset).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return 2

    csvv = validate_csv(csv_path, dry_run=args.dry_run)
    repo = discover_repo_expectations(repo_root, args.index_kind, verbose=args.verbose)
    verdict = build_verdict(csvv, repo)

    print_report(csvv, repo, verdict, verbose=args.verbose)
    return 0 if verdict.directly_compatible and csvv.passed else 1


if __name__ == "__main__":
    sys.exit(main())
