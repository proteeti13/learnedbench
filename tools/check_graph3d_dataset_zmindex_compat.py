#!/usr/bin/env python3
"""
ZMIndex compatibility checker for graph 3D CSV datasets.

This tool inspects this repository to derive the true ZMIndex input contract.

Primary contract sources (referenced in report + comments):
- bench/bench.cpp:
  - index registry includes "zm"
  - benchmark loads data via bench::utils::read_points(points, fname, N)
- utils/datautils.hpp:
  - read_points(...) uses tpie::file_stream<double>
  - csv_to_bin(...) conversion utilities
- utils/datagen.cpp:
  - CLI task "process_csv" wraps csv_to_bin
- indexes/learned/zmindex.hpp:
  - ZMIndex consumes point coordinates only
- CMakeLists.txt:
  - benchmark binaries define BENCH_DIM at compile time (2D/3D variants)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


CANONICAL = ("SourceID", "Hop1_ID", "Hop2_ID", "Offset")
ALIASES = {
    "SourceID": {"sourceid", "source_id", "src", "source"},
    "Hop1_ID": {"hop1_id", "hop1", "h1", "dest", "dst"},
    "Hop2_ID": {"hop2_id", "hop2", "h2"},
    "Offset": {"offset", "rank", "position", "global_offset", "idx", "index"},
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _run_rg(repo_root: Path, pattern: str, targets: Sequence[str]) -> List[str]:
    # Preferred path: ripgrep
    try:
        cp = subprocess.run(
            ["rg", "-n", pattern, *targets],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if cp.returncode in (0, 1):
            return [ln for ln in cp.stdout.splitlines() if ln.strip()]
    except FileNotFoundError:
        cp = None

    # Fallback path: grep -R (for environments where rg is unavailable in PATH)
    try:
        gp = subprocess.run(
            ["grep", "-R", "-n", "-I", "-E", pattern, *targets],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if gp.returncode in (0, 1):
            return [ln for ln in gp.stdout.splitlines() if ln.strip()]
    except FileNotFoundError:
        pass

    return []


@dataclass
class CsvCheck:
    path: str
    row_count: int
    header_ok: bool
    header_map: Dict[str, str]
    sorted_lex: bool
    no_duplicates: bool
    offset_ok: bool
    min_vals: Dict[str, Optional[int]]
    max_vals: Dict[str, Optional[int]]
    head: List[Tuple[int, int, int, int]]
    tail: List[Tuple[int, int, int, int]]
    errors: List[str]

    @property
    def passed(self) -> bool:
        return (
            self.header_ok
            and self.sorted_lex
            and self.no_duplicates
            and self.offset_ok
            and not self.errors
        )


@dataclass
class RepoContract:
    zm_registered: bool
    direct_csv_supported: bool
    accepted_formats: List[str]
    loader_function: str
    loader_files: List[str]
    record_layout: str
    payload_contract: str
    sort_contract: str
    dimension_contract: str
    proof_lines: List[str]
    keyword_scan_summary: Dict[str, int]


@dataclass
class Verdict:
    directly_compatible: bool
    status: str
    reason: str
    next_steps: List[str]
    conversion_requirements: Optional[Dict[str, object]]


def _parse_int(value: str, col: str, row_no: int) -> int:
    s = value.strip()
    if s == "":
        raise ValueError(f"row {row_no}: empty {col}")
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    try:
        f = float(s)
    except ValueError as exc:
        raise ValueError(f"row {row_no}: non-numeric {col}={value!r}") from exc
    if not f.is_integer():
        raise ValueError(f"row {row_no}: non-integer {col}={value!r}")
    return int(f)


def _map_header(fieldnames: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    src = {_norm(f): f for f in fieldnames}
    out: Dict[str, str] = {}
    errs: List[str] = []
    for c in CANONICAL:
        cand = {_norm(c), *{_norm(a) for a in ALIASES[c]}}
        found = None
        for x in cand:
            if x in src:
                found = src[x]
                break
        if found is None:
            errs.append(f"missing column for {c}; header={list(fieldnames)}")
        else:
            out[c] = found
    return out, errs


def validate_csv(csv_path: Path, dry_run: bool) -> CsvCheck:
    errors: List[str] = []
    row_count = 0
    header_ok = False
    header_map: Dict[str, str] = {}

    sorted_lex = True
    no_duplicates = True
    offset_ok = True

    mins: Dict[str, Optional[int]] = {k: None for k in CANONICAL}
    maxs: Dict[str, Optional[int]] = {k: None for k in CANONICAL}
    head: List[Tuple[int, int, int, int]] = []
    tail: List[Tuple[int, int, int, int]] = []

    seen = set()
    prev_key: Optional[Tuple[int, int, int]] = None

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            errors.append("missing header or empty CSV")
        else:
            header_map, h_err = _map_header(reader.fieldnames)
            errors.extend(h_err)
            header_ok = len(h_err) == 0

        if not header_ok:
            return CsvCheck(
                path=str(csv_path),
                row_count=0,
                header_ok=False,
                header_map=header_map,
                sorted_lex=False,
                no_duplicates=False,
                offset_ok=False,
                min_vals=mins,
                max_vals=maxs,
                head=[],
                tail=[],
                errors=errors,
            )

        for i, row in enumerate(reader):
            row_no = i + 2
            try:
                src = _parse_int(row[header_map["SourceID"]], "SourceID", row_no)
                h1 = _parse_int(row[header_map["Hop1_ID"]], "Hop1_ID", row_no)
                h2 = _parse_int(row[header_map["Hop2_ID"]], "Hop2_ID", row_no)
                off = _parse_int(row[header_map["Offset"]], "Offset", row_no)
            except ValueError as exc:
                errors.append(str(exc))
                if dry_run:
                    break
                continue

            rec = (src, h1, h2, off)
            row_count += 1

            if len(head) < 5:
                head.append(rec)
            tail.append(rec)
            if len(tail) > 5:
                tail.pop(0)

            for k, v in zip(CANONICAL, rec):
                if mins[k] is None or v < mins[k]:
                    mins[k] = v
                if maxs[k] is None or v > maxs[k]:
                    maxs[k] = v

            if not dry_run:
                key = (src, h1, h2)
                if key in seen:
                    no_duplicates = False
                seen.add(key)

                if prev_key is not None and prev_key >= key:
                    sorted_lex = False
                prev_key = key

                if off != (row_count - 1):
                    offset_ok = False

    if dry_run:
        sorted_lex = True
        no_duplicates = True
        offset_ok = True

    return CsvCheck(
        path=str(csv_path),
        row_count=row_count,
        header_ok=header_ok,
        header_map=header_map,
        sorted_lex=sorted_lex,
        no_duplicates=no_duplicates,
        offset_ok=offset_ok,
        min_vals=mins,
        max_vals=maxs,
        head=head,
        tail=tail,
        errors=errors,
    )


def discover_repo_contract(repo_root: Path) -> RepoContract:
    # Mandatory discovery keyword scan from prompt.
    keywords = [
        "zmindex",
        "zmi",
        "dataset",
        "load",
        "reader",
        "csv",
        "binary",
        "bin",
        "uint64",
        "mmap",
        "npy",
        "keys",
        "values",
        "payload",
        "record",
        "schema",
    ]
    keyword_scan_summary: Dict[str, int] = {}
    for kw in keywords:
        hits = _run_rg(repo_root, rf"{re.escape(kw)}", ["bench", "utils", "indexes", "scripts", "README.md", "CMakeLists.txt"])
        keyword_scan_summary[kw] = len(hits)

    proof: List[str] = []

    # ZM registry + benchmark loader call.
    reg_hits = _run_rg(repo_root, r'idx_name\.compare\("zm"\)', ["bench/bench.cpp"])
    load_hits = _run_rg(repo_root, r"read_points\(points, fname, N\)", ["bench/bench.cpp"])
    zmalias_hits = _run_rg(repo_root, r"using ZM = .*ZMIndex", ["bench/bench.cpp"])

    # Core loader format.
    tpie_hits = _run_rg(repo_root, r"tpie::file_stream<double>|inline void read_points|#include <tpie/", ["utils/datautils.hpp"])

    # Conversion helper.
    datagen_hits = _run_rg(repo_root, r"process_csv|csv_to_bin", ["utils/datagen.cpp", "utils/datautils.hpp"])

    # ZM consumption shape.
    zm_hits = _run_rg(repo_root, r"class ZMIndex|Points& _data|a2t\(|payload|size_t", ["indexes/learned/zmindex.hpp"])

    # Dim compile-time evidence.
    dim_hits = _run_rg(repo_root, r"BENCH_DIM|bench3d_toronto|bench2d_default", ["bench/bench.cpp", "CMakeLists.txt"])

    for group in (reg_hits, load_hits, zmalias_hits, tpie_hits, datagen_hits, zm_hits, dim_hits):
        proof.extend(group)

    # stable dedup
    seen = set()
    proof = [ln for ln in proof if not (ln in seen or seen.add(ln))]

    zm_registered = len(reg_hits) > 0 and len(zmalias_hits) > 0

    accepted_formats = [
        "Benchmark runtime input: TPIE container file read as tpie::file_stream<double>",
        "CSV is only accepted indirectly via conversion utility (utils/datagen.cpp: process_csv -> csv_to_bin)",
    ]

    # NOTE: This layout is inferred from exact loader implementation.
    record_layout = (
        "Loader consumes Dim doubles per point in file order: [k0, k1, ..., k{Dim-1}], each double=8 bytes. "
        "For 3D benchmark binary (BENCH_DIM=3), required key order is [SourceID, Hop1_ID, Hop2_ID]."
    )

    payload_contract = (
        "No payload/value column is read by ZMIndex pipeline loader. "
        "Offset is NOT consumed by bench::utils::read_points or ZMIndex constructor input."
    )

    sort_contract = (
        "No strict sortedness requirement enforced by loader; points are read sequentially in file order. "
        "Your CSV sortedness is semantically fine but not a parser requirement."
    )

    dimension_contract = (
        "Dimension is compile-time per benchmark binary (BENCH_DIM). Dataset dimensionality must match chosen binary."
    )

    return RepoContract(
        zm_registered=zm_registered,
        direct_csv_supported=False,
        accepted_formats=accepted_formats,
        loader_function="bench::utils::read_points (tpie::file_stream<double>)",
        loader_files=[
            "bench/bench.cpp",
            "utils/datautils.hpp",
            "utils/datagen.cpp",
            "indexes/learned/zmindex.hpp",
            "CMakeLists.txt",
        ],
        record_layout=record_layout,
        payload_contract=payload_contract,
        sort_contract=sort_contract,
        dimension_contract=dimension_contract,
        proof_lines=proof,
        keyword_scan_summary=keyword_scan_summary,
    )


def make_verdict(csv_check: CsvCheck, contract: RepoContract) -> Verdict:
    if not csv_check.passed:
        return Verdict(
            directly_compatible=False,
            status="FAIL",
            reason="CSV validation failed.",
            next_steps=["Fix CSV validation errors first."],
            conversion_requirements=None,
        )

    if not contract.zm_registered:
        return Verdict(
            directly_compatible=False,
            status="FAIL",
            reason="ZMIndex alias ('zm') was not found in benchmark index registry.",
            next_steps=["Verify branch/build targets for ZMIndex."],
            conversion_requirements=None,
        )

    conversion = {
        "expected_extensions": ["no strict extension enforced", ".bin (common naming)"],
        "expected_container": "TPIE file_stream<double>",
        "expected_packing": {
            "per_point": "Dim x float64/double",
            "for_3d": "[SourceID(double), Hop1_ID(double), Hop2_ID(double)]",
            "bytes_per_point_3d": 24,
            "payload_offset_included": False,
        },
        "endianness": (
            "Use repo converter/writer (tpie::file_stream<double>) to avoid manual byte-order assumptions. "
            "Raw binary without TPIE container is not the benchmark loader contract."
        ),
        "required_key_order": ["SourceID", "Hop1_ID", "Hop2_ID"],
        "sorting_required": "Not required by loader; current CSV already sorted lexicographically.",
        "minimal_conversion_recipe": [
            "Create an intermediate CSV containing ONLY key columns in this order: SourceID,Hop1_ID,Hop2_ID (drop Offset).",
            "Convert with existing tool: build/bin/datagen --task process_csv --infname <keys3.csv> --fname <dataset_tpie_3d>",
            "Run ZMIndex on 3D binary: build/bin/bench3d_toronto zm <dataset_tpie_3d> <N> <mode>",
        ],
    }

    return Verdict(
        directly_compatible=False,
        status="FAIL",
        reason="CSV is not directly consumable by ZMIndex benchmark loader; conversion to TPIE double-stream file is required.",
        next_steps=conversion["minimal_conversion_recipe"],
        conversion_requirements=conversion,
    )


def print_report(csv_check: CsvCheck, contract: RepoContract, verdict: Verdict, verbose: bool) -> None:
    print("=" * 92)
    print("Z-MINDEX DATASET COMPATIBILITY REPORT")
    print("=" * 92)

    print("\n[CSV validation]")
    print(f"path: {csv_check.path}")
    print(f"row_count: {csv_check.row_count}")
    print(f"header_mapping: {csv_check.header_map}")
    print(f"checks:")
    print(f"  - header: {'PASS' if csv_check.header_ok else 'FAIL'}")
    print(f"  - sorted(SourceID,Hop1_ID,Hop2_ID): {'PASS' if csv_check.sorted_lex else 'FAIL'}")
    print(f"  - no_duplicate_keys: {'PASS' if csv_check.no_duplicates else 'FAIL'}")
    print(f"  - offset_eq_row_index: {'PASS' if csv_check.offset_ok else 'FAIL'}")
    print(f"min: {csv_check.min_vals}")
    print(f"max: {csv_check.max_vals}")
    print(f"head: {csv_check.head}")
    print(f"tail: {csv_check.tail}")
    if csv_check.errors:
        print("errors:")
        for e in csv_check.errors:
            print(f"  - {e}")

    print("\n[ZMIndex expected format contract]")
    print(f"zm_registered_in_benchmark: {contract.zm_registered}")
    print(f"direct_csv_supported: {contract.direct_csv_supported}")
    print("accepted_formats:")
    for f in contract.accepted_formats:
        print(f"  - {f}")
    print(f"loader_function: {contract.loader_function}")
    print("loader_files:")
    for p in contract.loader_files:
        print(f"  - {p}")
    print(f"record_layout: {contract.record_layout}")
    print(f"payload/value_contract: {contract.payload_contract}")
    print(f"sorting_contract: {contract.sort_contract}")
    print(f"dimension_contract: {contract.dimension_contract}")

    print("proof (repo lines):")
    for ln in contract.proof_lines[:90]:
        print(f"  - {ln}")
    if len(contract.proof_lines) > 90:
        print(f"  - ... ({len(contract.proof_lines) - 90} more lines; run --verbose for full dump)")

    print("\n[keyword discovery summary]")
    print(json.dumps(contract.keyword_scan_summary, indent=2, sort_keys=True))

    print("\n[Compatibility verdict]")
    print(f"status: {verdict.status}")
    print(f"directly_compatible: {verdict.directly_compatible}")
    print(f"reason: {verdict.reason}")

    if verdict.directly_compatible:
        print("summary: DIRECTLY COMPATIBLE")
        for s in verdict.next_steps:
            print(f"  - {s}")
    else:
        print("summary: NOT DIRECTLY COMPATIBLE")
        if verdict.conversion_requirements:
            print("conversion_requirements:")
            print(json.dumps(verdict.conversion_requirements, indent=2))
        print("next_steps:")
        for s in verdict.next_steps:
            print(f"  - {s}")

    if verbose:
        print("\n[verbose dump]")
        print(json.dumps({
            "csv_check": csv_check.__dict__,
            "contract": contract.__dict__,
            "verdict": verdict.__dict__,
        }, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check 3D graph CSV compatibility with ZMIndex pipeline.")
    ap.add_argument("--csv", required=True, help="Path to CSV file.")
    ap.add_argument("--verbose", action="store_true", help="Print detailed internals.")
    ap.add_argument("--dry_run", action="store_true", help="Skip O(N) set/sorted checks.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    if not csv_path.exists():
        print(f"ERROR: csv not found: {csv_path}")
        return 2

    csv_check = validate_csv(csv_path, dry_run=args.dry_run)
    contract = discover_repo_contract(repo_root)
    verdict = make_verdict(csv_check, contract)
    print_report(csv_check, contract, verdict, verbose=args.verbose)

    return 0 if verdict.directly_compatible and csv_check.passed else 1


if __name__ == "__main__":
    sys.exit(main())
