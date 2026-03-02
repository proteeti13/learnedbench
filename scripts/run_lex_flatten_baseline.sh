#!/usr/bin/env bash
# =============================================================================
# run_lex_flatten_baseline.sh – End-to-end LexFlattenBaseline Pipeline
# =============================================================================
# Experiment: LexFlattenBaseline (formerly "Path A")
# Index:      PGM-Index (1-D learned index on lexicographic key K)
# Dataset:    Dense 3-D graph CSV (SourceID;Hop1_ID;Hop2_ID;Offset)
#
# Steps:
#   1. (Optional) Generate CSV using the Python generator
#   2. Convert CSV → TPIE format for y_mode=const and y_mode=hop2
#   3. Run bench_lfb with each y_mode + report all thesis metrics
#   4. Run sensitivity comparison (proves y_mode irrelevant for 1-D PGM,
#      motivating True3DLearnedIndex / Path B)
#
# Usage (from repo root):
#   bash scripts/run_lex_flatten_baseline.sh [--n=<N>] [--queries=<Q>]
#                                            [--epsilon=<E>] [--csv=<path>]
#
# Defaults:
#   --n=10000   (matches the 10x10x100 generated dataset)
#   --queries=10000
#   --epsilon=64
#   --csv=tools/datasets/lex_flatten_baseline/data/dense_3d_with_offset.csv
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
BUILD_DIR="${REPO_ROOT}/build"
DATA_DIR="${BUILD_DIR}/data"

# ---- Defaults ---------------------------------------------------------------
N=10000
Q=10000
EPSILON=64
CSV_PATH="${REPO_ROOT}/tools/datasets/lex_flatten_baseline/data/dense_3d_with_offset.csv"

# ---- Parse args -------------------------------------------------------------
for arg in "$@"; do
  case "$arg" in
    --n=*)       N="${arg#--n=}"       ;;
    --queries=*) Q="${arg#--queries=}" ;;
    --epsilon=*) EPSILON="${arg#--epsilon=}" ;;
    --csv=*)     CSV_PATH="${arg#--csv=}" ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

TPIE_CONST="${DATA_DIR}/lex_flatten_baseline_${N}_const.tpie"
TPIE_HOP2="${DATA_DIR}/lex_flatten_baseline_${N}_hop2.tpie"
LOG_DIR="${BUILD_DIR}/logs/lex_flatten_baseline"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "======================================"
echo "  LexFlattenBaseline Pipeline"
echo "  REPO  : ${REPO_ROOT}"
echo "  N     : ${N}"
echo "  Q     : ${Q}"
echo "  eps   : ${EPSILON}"
echo "  CSV   : ${CSV_PATH}"
echo "======================================"

# ---- Sanity: ensure output dirs exist ---------------------------------------
mkdir -p "${DATA_DIR}" "${LOG_DIR}"

# ---- Step 1: Check for CSV --------------------------------------------------
if [ ! -f "${CSV_PATH}" ]; then
  echo "[1] CSV not found – generating with Python generator..."
  python3 "${REPO_ROOT}/tools/datasets/lex_flatten_baseline/generate_dense_3d.py"
  echo "[1] CSV generated: ${CSV_PATH}"
else
  echo "[1] CSV found: ${CSV_PATH}"
fi

# ---- Step 2: Build the executables ------------------------------------------
echo ""
echo "[2] Building csv_to_tpie_lfb and bench_lfb ..."
cd "${BUILD_DIR}"
make -j"$(nproc)" csv_to_tpie_lfb bench_lfb 2>&1 | tail -5
cd "${REPO_ROOT}"
echo "[2] Build complete."

CONVERTER="${BUILD_DIR}/bin/csv_to_tpie_lfb"
BENCHMARK="${BUILD_DIR}/bin/bench_lfb"

if [ ! -x "${CONVERTER}" ]; then
  echo "ERROR: ${CONVERTER} not found. Check build." && exit 1
fi
if [ ! -x "${BENCHMARK}" ]; then
  echo "ERROR: ${BENCHMARK} not found. Check build." && exit 1
fi

# ---- Step 3: Convert CSV → TPIE (both y_modes) ------------------------------
echo ""
echo "[3a] Converting CSV → TPIE (y_mode=const) ..."
"${CONVERTER}" "${CSV_PATH}" "${TPIE_CONST}" \
  --y_mode=const --verify \
  2>&1 | tee "${LOG_DIR}/convert_const_${TIMESTAMP}.log"

echo ""
echo "[3b] Converting CSV → TPIE (y_mode=hop2) ..."
"${CONVERTER}" "${CSV_PATH}" "${TPIE_HOP2}" \
  --y_mode=hop2 --verify \
  2>&1 | tee "${LOG_DIR}/convert_hop2_${TIMESTAMP}.log"

echo ""
echo "[3] TPIE files:"
ls -lh "${DATA_DIR}"/lex_flatten_baseline_*.tpie

# ---- Step 4: Run bench_lfb (y_mode=const) -----------------------------------
echo ""
echo "[4a] Running bench_lfb (y_mode=const, epsilon=${EPSILON}) ..."
"${BENCHMARK}" "${TPIE_CONST}" "${N}" \
  --queries="${Q}" \
  --epsilon="${EPSILON}" \
  --y_mode=const \
  2>&1 | tee "${LOG_DIR}/bench_const_${TIMESTAMP}.log"

# ---- Step 5: Run bench_lfb (y_mode=hop2) ------------------------------------
echo ""
echo "[4b] Running bench_lfb (y_mode=hop2, epsilon=${EPSILON}) ..."
"${BENCHMARK}" "${TPIE_HOP2}" "${N}" \
  --queries="${Q}" \
  --epsilon="${EPSILON}" \
  --y_mode=hop2 \
  2>&1 | tee "${LOG_DIR}/bench_hop2_${TIMESTAMP}.log"

# ---- Step 6: Sensitivity comparison -----------------------------------------
# Passes the _const file with --sensitivity; bench_lfb internally substitutes
# _const → _hop2 in the filename to load the second variant.
echo ""
echo "[5] Running sensitivity comparison (const vs hop2) ..."
"${BENCHMARK}" "${TPIE_CONST}" "${N}" \
  --queries="${Q}" \
  --epsilon="${EPSILON}" \
  --y_mode=const \
  --sensitivity \
  2>&1 | tee "${LOG_DIR}/bench_sensitivity_${TIMESTAMP}.log"

echo ""
echo "======================================"
echo "  Pipeline complete."
echo "  Logs  : ${LOG_DIR}/"
echo "  Data  : ${DATA_DIR}/"
echo "======================================"

# ---- Optional: bench_rsmi point_lookup (requires RSMI built with -DRSMI=ON) -
BENCH_RSMI="${BUILD_DIR}/bin/bench_rsmi"
if [ -x "${BENCH_RSMI}" ]; then
  echo ""
  echo "[BONUS] bench_rsmi found – running LexFlattenBaseline point_lookup ..."
  "${BENCH_RSMI}" rsmi "${TPIE_CONST}" "${N}" point_lookup "${Q}" 42 \
    2>&1 | tee "${LOG_DIR}/bench_rsmi_lfb_${TIMESTAMP}.log"
else
  echo ""
  echo "[INFO] bench_rsmi not found (requires -DRSMI=ON + PyTorch/libtorch)."
  echo "       When available, run:"
  echo "       ${BENCH_RSMI} rsmi ${TPIE_CONST} ${N} point_lookup ${Q} 42"
fi
