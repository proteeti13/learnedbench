#!/usr/bin/env bash
# run_zmindex_all.sh — Run ZM-Index benchmark on all 8 thesis datasets × 3 scales (24 runs)
#
# Run from repo root:
#   bash scripts/run_zmindex_all.sh
# Or inside tmux for SSH resilience:
#   tmux new-session -d -s zmall 'bash scripts/run_zmindex_all.sh 2>&1 | tee build/results/zmindex_all.log'

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO}/build/bin/bench_zmindex_all"
DATA_RSMI="/home/proteeti/RSMI/datasets"
RESULTS="${REPO}/build/results/zmindex_per_run"
QUERIES=100000
SEED=42

# Verify binary
if [ ! -f "${BIN}" ]; then
    echo "ERROR: binary not found at ${BIN}"
    echo "Build with:"
    echo "  mkdir -p build/bin && g++ -O3 -DNDEBUG -std=c++17 -mbmi2 \\"
    echo "    -I\${REPO} -I\${REPO}/indexes -I\${REPO}/utils \\"
    echo "    -I\${REPO}/tpie_stub -I/usr/local/boost/include -DBENCH_DIM=3 \\"
    echo "    bench/bench_zmindex_all.cpp -o build/bin/bench_zmindex_all -pthread"
    exit 1
fi

mkdir -p "${RESULTS}"

run_bench() {
    local dataset_name="$1"
    local data_source="$2"
    local distribution="$3"
    local graph_type="$4"
    local full_size="$5"
    local txt_file="$6"
    local N="$7"
    local label="$8"   # used in output filename

    local out_csv="${RESULTS}/zm_${label}.csv"

    if [ ! -f "${txt_file}" ]; then
        echo "  SKIP: data file not found: ${txt_file}"
        return
    fi

    echo ""
    echo "================================================================"
    echo "  ${dataset_name}  N=${N}  (${label})"
    echo "================================================================"
    "${BIN}" \
        "${txt_file}" "${N}" \
        "${dataset_name}" "${data_source}" "${distribution}" "${graph_type}" \
        "${full_size}" \
        --queries "${QUERIES}" \
        --seed "${SEED}" \
        --out_csv "${out_csv}"
}

echo "========================================================"
echo "  ZM-Index All-Datasets Benchmark — 24 runs"
echo "  Seed=${SEED}  Queries=${QUERIES}"
echo "========================================================"
echo ""

# ── 1. wiki_vote (real, directed, voting_network) ────────────────────────────
run_bench wiki_vote SNAP "real_graph_directed" voting_network 4542805 \
    "${DATA_RSMI}/wiki_vote_triples.txt" 1000000    "wiki_vote_1m"
run_bench wiki_vote SNAP "real_graph_directed" voting_network 4542805 \
    "${DATA_RSMI}/wiki_vote_triples.txt" 2500000    "wiki_vote_2500k"
run_bench wiki_vote SNAP "real_graph_directed" voting_network 4542805 \
    "${DATA_RSMI}/wiki_vote_triples.txt" 4542805    "wiki_vote_full"

# ── 2. roadnet_ca (real, undirected, road_network) ───────────────────────────
run_bench roadnet_ca SNAP "real_graph_undirected" road_network 17523394 \
    "${DATA_RSMI}/roadnet_ca_triples.txt" 1000000   "roadnet_ca_1m"
run_bench roadnet_ca SNAP "real_graph_undirected" road_network 17523394 \
    "${DATA_RSMI}/roadnet_ca_triples.txt" 2500000   "roadnet_ca_2500k"
run_bench roadnet_ca SNAP "real_graph_undirected" road_network 17523394 \
    "${DATA_RSMI}/roadnet_ca_triples.txt" 17523394  "roadnet_ca_full"

# ── 3. web_google (real, directed, web_graph) ────────────────────────────────
# NOTE: full web_google is 60M rows — flagged as potentially slow.
run_bench web_google SNAP "real_graph_directed" web_graph 60687836 \
    "${DATA_RSMI}/web_google_triples.txt" 1000000   "web_google_1m"
run_bench web_google SNAP "real_graph_directed" web_graph 60687836 \
    "${DATA_RSMI}/web_google_triples.txt" 2500000   "web_google_2500k"
run_bench web_google SNAP "real_graph_directed" web_graph 60687836 \
    "${DATA_RSMI}/web_google_triples.txt" 60687836  "web_google_full"

# ── 4. uniform_sparse  [1, 10000000) ─────────────────────────────────────────
run_bench uniform_sparse Synthetic "uniform([1,10000000))" N/A 60000000 \
    "${DATA_RSMI}/uniform_sparse_60M.txt" 1000000   "uniform_sparse_1m"
run_bench uniform_sparse Synthetic "uniform([1,10000000))" N/A 60000000 \
    "${DATA_RSMI}/uniform_sparse_60M.txt" 5000000   "uniform_sparse_5m"
run_bench uniform_sparse Synthetic "uniform([1,10000000))" N/A 60000000 \
    "${DATA_RSMI}/uniform_sparse_60M.txt" 10000000  "uniform_sparse_10m"

# ── 5. uniform_dense  [1, 500000) ────────────────────────────────────────────
run_bench uniform_dense Synthetic "uniform([1,500000))" N/A 60000000 \
    "${DATA_RSMI}/uniform_dense_60M.txt" 1000000    "uniform_dense_1m"
run_bench uniform_dense Synthetic "uniform([1,500000))" N/A 60000000 \
    "${DATA_RSMI}/uniform_dense_60M.txt" 5000000    "uniform_dense_5m"
run_bench uniform_dense Synthetic "uniform([1,500000))" N/A 60000000 \
    "${DATA_RSMI}/uniform_dense_60M.txt" 10000000   "uniform_dense_10m"

# ── 6. uniform_matched  [1, 1000000) ─────────────────────────────────────────
run_bench uniform_matched Synthetic "uniform([1,1000000))" N/A 60000000 \
    "${DATA_RSMI}/uniform_matched_60M.txt" 1000000  "uniform_matched_1m"
run_bench uniform_matched Synthetic "uniform([1,1000000))" N/A 60000000 \
    "${DATA_RSMI}/uniform_matched_60M.txt" 5000000  "uniform_matched_5m"
run_bench uniform_matched Synthetic "uniform([1,1000000))" N/A 60000000 \
    "${DATA_RSMI}/uniform_matched_60M.txt" 10000000 "uniform_matched_10m"

# ── 7. normal  N(500000, 166667²) ────────────────────────────────────────────
run_bench normal Synthetic "normal(mu=500000 sigma=166667)" N/A 60000000 \
    "${DATA_RSMI}/normal_60M.txt" 1000000            "normal_1m"
run_bench normal Synthetic "normal(mu=500000 sigma=166667)" N/A 60000000 \
    "${DATA_RSMI}/normal_60M.txt" 5000000            "normal_5m"
run_bench normal Synthetic "normal(mu=500000 sigma=166667)" N/A 60000000 \
    "${DATA_RSMI}/normal_60M.txt" 10000000           "normal_10m"

# ── 8. lognormal ─────────────────────────────────────────────────────────────
run_bench lognormal Synthetic "lognormal(mu=0.0 sigma=1.0 scale=1M/P99)" N/A 60000000 \
    "${DATA_RSMI}/lognormal_60M.txt" 1000000         "lognormal_1m"
run_bench lognormal Synthetic "lognormal(mu=0.0 sigma=1.0 scale=1M/P99)" N/A 60000000 \
    "${DATA_RSMI}/lognormal_60M.txt" 5000000         "lognormal_5m"
run_bench lognormal Synthetic "lognormal(mu=0.0 sigma=1.0 scale=1M/P99)" N/A 60000000 \
    "${DATA_RSMI}/lognormal_60M.txt" 10000000        "lognormal_10m"

echo ""
echo "========================================================"
echo "  All 24 runs complete."
echo "  Per-run CSVs saved to: ${RESULTS}/"
echo "  Next: python3 scripts/combine_zmindex_results.py"
echo "========================================================"
