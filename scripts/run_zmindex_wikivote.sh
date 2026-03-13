#!/usr/bin/env bash
# run_zmindex_wikivote.sh — End-to-end ZM-Index benchmark on Wiki-Vote 1M
#
# Prereqs
#   - build/data/wiki_1m_points.tpie       (from the Flood benchmark run)
#   - build/data/wiki_1m_queries_100k.bin  (same)
#   Both files are already present from prior experiments.
#
# Build (from repo root):
#   cd build && cmake .. -DRSMI=OFF && make bench_zmindex_wv -j$(nproc)
#
# Run this script from the repo root:
#   bash scripts/run_zmindex_wikivote.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_ROOT}/build/bin"
DATA="${REPO_ROOT}/build/data"
RESULTS="${REPO_ROOT}/build/results"

mkdir -p "${RESULTS}"

TPIE_FILE="${DATA}/wiki_1m_points.tpie"
QUERY_FILE="${DATA}/wiki_1m_queries_100k.bin"
N=1000000
Q=100000
OUT_CSV="${RESULTS}/zmindex_wikivote.csv"

echo "==================================================="
echo "ZM-Index Wiki-Vote benchmark"
echo "  Data    : ${TPIE_FILE}"
echo "  Queries : ${QUERY_FILE}"
echo "  N=${N}  Q=${Q}"
echo "==================================================="

# ── Check prerequisites ──────────────────────────────────────────────────────
if [ ! -f "${TPIE_FILE}" ]; then
    echo "ERROR: ${TPIE_FILE} not found."
    echo "  Run the Flood wiki benchmark first to generate it, or generate via:"
    echo "  python3 tools/datasets/wiki_vote/prepare_wikivote.py"
    exit 1
fi
if [ ! -f "${QUERY_FILE}" ]; then
    echo "ERROR: ${QUERY_FILE} not found."
    exit 1
fi
if [ ! -f "${BIN}/bench_zmindex_wv" ]; then
    echo "ERROR: ${BIN}/bench_zmindex_wv not found. Build first:"
    echo "  cd build && cmake .. -DRSMI=OFF && make bench_zmindex_wv -j\$(nproc)"
    exit 1
fi

# ── Run benchmark ────────────────────────────────────────────────────────────
echo ""
"${BIN}/bench_zmindex_wv" \
    "${TPIE_FILE}" \
    "${QUERY_FILE}" \
    "${N}" \
    --queries "${Q}" \
    --out_csv "${OUT_CSV}"

echo ""
echo "Done. Results in: ${OUT_CSV}"
