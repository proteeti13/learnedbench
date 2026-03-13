---
name: ZMIndex Wiki-Vote Experiment
description: ZM-Index point-lookup benchmark on Wiki-Vote 1M dataset — thesis Experiment 5 (flattened-1D baseline)
type: project
---

ZM-Index benchmarked on Wiki-Vote 1M as flattened-1D baseline vs true-3D learned indexes (RSMI3D, LISA, Flood).

**Why:** Thesis comparison: Morton-curve linearisation (ZMIndex) vs true-3D learned indexing. Same dataset, same Q=100K workload.

**How to apply:** Use these results in the thesis comparison table alongside RSMI3D (true-3D) results.

## Files created/modified

| File | Change |
|------|--------|
| `indexes/pgm/pgm_index_variants.hpp` | Added `point_query()` to `MultidimensionalPGMIndex` — returns `{found, pgm_window}` |
| `indexes/learned/zmindex.hpp` | Added `PointQueryResult` struct + `point_lookup(Point&)` method |
| `bench/bench_zmindex_wv.cpp` | New standalone benchmark: TPIE load + ZMIndex<3,64> + 100K point lookups |
| `CMakeLists.txt` | Added `bench_zmindex_wv` target (-mbmi2, TPIE+pthread only); fixed ANN_PATH bug |
| `scripts/run_zmindex_wikivote.sh` | End-to-end pipeline script |

## Data files used (already existed)
- `build/data/wiki_1m_points.tpie` — 1M × 3 doubles (TPIE format)
- `build/data/wiki_1m_queries_100k.bin` — 100K × 3 doubles (raw binary, actual 3D coordinates)

## Results (N=1M, Q=100K, Epsilon=64, Grid=99^3)

| Metric | Value |
|--------|-------|
| Build Time (s) | 0.0759 |
| Index Size (MB) | 7.66 |
| Mean Lookup Latency (us) | 0.1509 |
| P95 Lookup Latency (us) | 0.2510 |
| Query Throughput (q/s) | 6,625,332 |
| Correctness (%) | 100.00 |
| Avg PGM Refine Window | 129.94 (≈ 2×64+2=130) |

CSV saved: `build/results/zmindex_wikivote.csv`

## Key design decisions

1. **Point lookup** = `ZMIndex::point_lookup(p)` which calls `MultidimensionalPGMIndex::point_query(a2t(p))`:
   - `a2t(p)` → grid cell IDs (0..98 per dim, resolution=99)
   - Morton encode → uint64 code
   - PGM search → [lo, hi) window (size ≈ 130)
   - Binary search in [lo, hi) for exact Morton code match

2. **Correctness** = 100% — queries are from dataset; Morton code always found in PGM window.

3. **Offset column** — wiki_vote_triples.txt offsets are LEX-rank; ZMIndex uses MORTON-rank internally. The offset column is NOT used — correctness is measured by Morton-code retrieval, not lex-offset prediction.

4. **Query workload** — reuses `wiki_1m_queries_100k.bin` from prior Flood experiment (100K actual 3D coordinates from the 1M dataset).

## Build & run commands
```bash
# Build (from repo root)
cd build && cmake .. -DRSMI=OFF && make bench_zmindex_wv -j$(nproc)

# Run
build/bin/bench_zmindex_wv \
  build/data/wiki_1m_points.tpie \
  build/data/wiki_1m_queries_100k.bin \
  1000000 --queries 100000 \
  --out_csv build/results/zmindex_wikivote.csv

# Or use the script:
bash scripts/run_zmindex_wikivote.sh
```
