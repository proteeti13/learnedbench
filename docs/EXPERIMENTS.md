# Learnedbench — Experiment Documentation

This file documents all experiments in the repo, how to reproduce them, and
what their results mean.

---

## Experiment 1 — LexFlattenBaseline (10K dense 3D, PGM / ZMI)

**What it tests**: flattening 3D graph-path keys (SourceID, Hop1_ID, Hop2_ID)
into a 1-D lexicographic rank and indexing with the Z-curve Map Index (ZMI)
backed by a PGM model.  The benchmark verifies point-lookup correctness and
measures PGM prediction error / refinement cost.

**Dataset**: 10 × 10 × 100 = 10,000 dense grid, two variants:
- `const` — Offset = trivial linear function (tests ideal case)
- `hop2`  — Offset = SourceID·1000 + Hop1_ID·100 + Hop2_ID

**Files**:
| File | Role |
|------|------|
| `tools/datasets/lex_flatten_baseline/generate_dense_3d.py` | generates CSV |
| `build/bin/csv_to_tpie_lfb` | CSV → TPIE converter |
| `build/bin/bench_lfb` | benchmark binary |
| `build/data/lex_flatten_baseline_10000_const.tpie` | data file |

**Run**:
```bash
# (re-)generate dataset
python3 tools/datasets/lex_flatten_baseline/generate_dense_3d.py

# convert to TPIE
build/bin/csv_to_tpie_lfb \
  tools/datasets/lex_flatten_baseline/data/dense_3d_with_offset.csv \
  build/data/lex_flatten_baseline_10000_const.tpie \
  --y_mode=const --verify

# benchmark
build/bin/bench_lfb \
  build/data/lex_flatten_baseline_10000_const.tpie 10000 \
  --queries=10000 --epsilon=64 --y_mode=const
```

**Reference results** (ε=64, Q=10K):

| Metric | const | hop2 |
|--------|-------|------|
| Build time (ms) | ~1 | ~1 |
| Index size (bytes) | 48 | 48 |
| Latency mean (ns) | ~45 | ~45 |
| Latency p99 (ns) | 61 | 61 |
| PredError(avg) | 15.43 | 15.43 |
| PredError(p99) | 31 | 31 |
| CorrSteps(avg) | ~8 | ~8 |
| Correctness | 100% | 100% |

**Takeaway**: `const` and `hop2` are identical — confirms the 3D structure
is invisible to a 1-D lex-flatten approach, motivating a true 3D index.

---

## Experiment 2 — RSMI3D on 1M Dense (ZMI-comparable)

**What it tests**: a true 3D learned index built on the full 3D coordinate
space of 1,000,000 dense graph-path keys.

**Architecture** (`indexes/rsmi3d/rsmi3d.hpp`):
- RSMI-style recursive spatial model index, *without* neural networks or
  libtorch (compatible with the Linux/WSL2 build environment).
- Uniform grid partitioning: `fanout^3` cells (default `10^3 = 1000` cells).
- Each cell holds points sorted by their global 3D Morton (Z-order) code.
- A PGM model (ε = 64) is trained per non-empty cell on `morton → rank`.
- Point lookup: cell dispatch → PGM predict → bounded binary search.

**Dataset**: 100 × 100 × 100 = 1,000,000 dense grid.
```
SourceID  ∈ [0, 99]
Hop1_ID   ∈ [0, 99]
Hop2_ID   ∈ [0, 99]
Offset    = SourceID × 10000 + Hop1_ID × 100 + Hop2_ID   (lex rank)
```

**Files**:
| File | Role |
|------|------|
| `tools/datasets/true_3d_learned_index/gen_true3d_1m.py` | Python data generator |
| `tools/datasets/true_3d_learned_index/raw_to_tpie.cpp` | raw doubles → TPIE converter |
| `build/bin/raw_to_tpie` | compiled converter |
| `indexes/rsmi3d/rsmi3d.hpp` | index implementation (header-only) |
| `bench/bench_rsmi3d.cpp` | benchmark source |
| `build/bin/bench_rsmi3d` | benchmark binary |
| `build/data/true3d_1m_points.raw` | generated: raw float64 points |
| `build/data/true3d_1m_points.tpie` | generated: TPIE points |
| `build/data/true3d_1m_offsets.bin` | generated: raw uint64 offsets |

### A) Build the new binaries

```bash
cd build
cmake .. -DRSMI=OFF   # only needed if CMakeLists.txt changed
make raw_to_tpie bench_rsmi3d -j$(nproc)
cd ..
```

### B) Generate the 1M dataset

```bash
# Step 1: generate raw files (Python, ~5s for 1M rows)
python3 tools/datasets/true_3d_learned_index/gen_true3d_1m.py \
  --out_dir build/data \
  --verify

# Step 2: convert points to TPIE format (~2s)
build/bin/raw_to_tpie \
  build/data/true3d_1m_points.raw \
  build/data/true3d_1m_points.tpie
```

Expected output of step 1:
```
============================================================
True3D 1M Dataset Generator
  Grid     : 100 x 100 x 100 = 1,000,000 points
  SourceID : [0, 99]
  Hop1_ID  : [0, 99]
  Hop2_ID  : [0, 99]
  Offset   = SourceID*10000 + Hop1_ID*100 + Hop2_ID
  Points   -> build/data/true3d_1m_points.raw  (24.0 MB)
  Offsets  -> build/data/true3d_1m_offsets.bin  (8.0 MB)
...
All verifications PASSED.
```

### C) Run the benchmark (recommended parameters)

```bash
build/bin/bench_rsmi3d \
  build/data/true3d_1m_points.tpie \
  build/data/true3d_1m_offsets.bin \
  1000000 \
  --queries 10000 \
  --seed 42 \
  --fanout 10
```

### D) Sample output format

```
====================================
Construct RSMI3D Epsilon=64
Dataset     : build/data/true3d_1m_points.tpie
N           : 1000000
Fanout      : 10  (10x10x10 = 1000 cells)
Queries     : 10000  (seed=42)
------------------------------------
Loading points from TPIE ...
Loading offsets ...
Build Time: <X> [ms]
Index Size: <X> Bytes
Non-empty cells: 1000 / 1000
------------------------------------
PointLookupStats:
PredError(avg)=<X>, PredError(max)=<X>
CorrSteps(avg)=<X>, CorrSteps(max)=<X>
FallbackRate=N/A (no explicit fallback path)
------------------------------------
ExtendedStats:
  Latency(mean)=<X> ns  Latency(p50)=<X> ns  Latency(p95)=<X> ns  Latency(p99)=<X> ns
  PredError(p50)=<X>  PredError(p95)=<X>  PredError(p99)=<X>
  CorrSteps(p95)=<X>  CorrSteps(p99)=<X>
  Correctness=10000/10000 (100.00%)
  Throughput=<X> queries/sec
====================================
```

### E) Knob sweep (optional)

```bash
# Vary fanout: fewer/larger cells
for F in 5 10 20; do
  echo "--- fanout=$F ---"
  build/bin/bench_rsmi3d \
    build/data/true3d_1m_points.tpie \
    build/data/true3d_1m_offsets.bin \
    1000000 --queries 10000 --seed 42 --fanout $F
done
```

### F) RSMI3D-specific knobs reported

| Knob | CLI flag | Default | Effect |
|------|----------|---------|--------|
| Grid fanout | `--fanout F` | 10 | cells per dim (F³ total cells, ~N/F³ pts/cell) |
| PGM epsilon | `RSMI3D_EPSILON` (compile-time) | 64 | refinement window = 2ε+2 |
| Dataset size | positional `N` | — | number of points to load |
| Queries | `--queries Q` | 10000 | number of point lookups |
| Seed | `--seed S` | 42 | random seed for query sampling |

**Metric correspondence with ZMI** (`bench3d_toronto`):

| Metric | ZMI output | RSMI3D output |
|--------|-----------|---------------|
| Build time | `Build Time: X [ms]` | `Build Time: X [ms]` |
| Index size | `Index Size: X Bytes` | `Index Size: X Bytes` |
| Pred error avg | `PredError(avg)=X` | `PredError(avg)=X` |
| Pred error max | `PredError(max)=X` | `PredError(max)=X` |
| Corr steps avg | `CorrSteps(avg)=X` | `CorrSteps(avg)=X` |
| Corr steps max | `CorrSteps(max)=X` | `CorrSteps(max)=X` |
| Latency p50/p95/p99 | (not in ZMI) | `ExtendedStats` section |
| Correctness | implicit | explicit percentage |
| Throughput | (not in ZMI) | `Throughput=X queries/sec` |

---

## Experiment 3 — RSMI3D: Dense vs Sparse 1M (true-3D benchmark)

**What it tests**: Compares RSMI3D on two 1M 3D datasets — a fully dense
grid (100×100×100) vs. a sparse random sample from [0,999]³ — using the
`bench_rsmi3d_true3d` binary with extended metrics including RefineWindow.

**Datasets**:
| Name | Description | Coordinate range | Size |
|------|-------------|-----------------|------|
| Dense3D_1M | 100×100×100 full lexicographic grid | [0,99]³ | 1,000,000 pts |
| Sparse3D_1M | 1M unique random triples from [0,999]³ (seed=42) | [0,999]³ | 1,000,000 pts |

**Files**:
| File | Role |
|------|------|
| `tools/datasets/true_3d_learned_index/gen_datasets_1m.py` | Python generator for both datasets |
| `build/bin/raw_to_tpie` | raw float64 → TPIE converter |
| `indexes/rsmi3d/rsmi3d.hpp` | RSMI3D index (recursive tree, PGM/LINEAR/MIDPOINT per leaf) |
| `bench/bench_rsmi3d_true3d.cpp` | benchmark source (100K queries, extended metrics) |
| `build/bin/bench_rsmi3d_true3d` | benchmark binary |
| `build/data/dense3d_1m_points.tpie` | Dense3D TPIE points |
| `build/data/dense3d_1m_offsets.bin` | Dense3D uint64 offsets |
| `build/data/sparse3d_1m_points.tpie` | Sparse3D TPIE points |
| `build/data/sparse3d_1m_offsets.bin` | Sparse3D uint64 offsets |

### A) Build

```bash
cd build
cmake .. -DRSMI=OFF
make bench_rsmi3d_true3d raw_to_tpie -j$(nproc)
cd ..
```

### B) Generate datasets

```bash
# Generate both dense and sparse raw files (~10s)
python3 tools/datasets/true_3d_learned_index/gen_datasets_1m.py \
  --datasets both --out_dir build/data --verify

# Convert to TPIE format
build/bin/raw_to_tpie build/data/dense3d_1m_points.raw  build/data/dense3d_1m_points.tpie
build/bin/raw_to_tpie build/data/sparse3d_1m_points.raw build/data/sparse3d_1m_points.tpie
```

### C) Run benchmarks

```bash
# Dense3D_1M
build/bin/bench_rsmi3d_true3d \
  build/data/dense3d_1m_points.tpie \
  build/data/dense3d_1m_offsets.bin \
  1000000 Dense3D_1M \
  --queries 100000 --seed 42 --fanout 10 --max_depth 1 --min_leaf 100 --model pgm

# Sparse3D_1M
build/bin/bench_rsmi3d_true3d \
  build/data/sparse3d_1m_points.tpie \
  build/data/sparse3d_1m_offsets.bin \
  1000000 Sparse3D_1M \
  --queries 100000 --seed 42 --fanout 10 --max_depth 1 --min_leaf 100 --model pgm
```

### D) Reference results (ε=64, Q=100K, fanout=10, max_depth=1)

| Metric | Dense3D_1M | Sparse3D_1M |
|--------|-----------|------------|
| Build time (ms) | ~100 | ~111 |
| Index size (bytes) | 16,400,656 (~15.6 MB) | 16,401,472 (~15.6 MB) |
| Non-empty leaves | 1000 / 1000 | 1000 / 1000 |
| Latency mean (ns) | ~319 | ~316 |
| Latency p50 (ns) | ~301 | ~290 |
| Latency p95 (ns) | ~539 | ~539 |
| Latency p99 (ns) | ~756 | ~788 |
| Throughput (queries/sec) | ~3.1M | ~3.2M |
| PredError mean | ~30.0 | ~30.1 |
| PredError p95 | 60 | 60 |
| PredError max | 65 | 65 |
| RefineWindow mean | ~125.5 | ~125.6 |
| RefineWindow p99 | 130 | 130 |
| CorrSteps(avg) | ~6.98 | ~6.98 |
| CorrSteps(max) | 8 | 8 |
| Correctness | **100%** | **100%** |

**Takeaway**: Dense and Sparse 1M datasets show nearly identical RSMI3D
performance — the uniform grid partitioning distributes both evenly across
1000 cells (~1000 pts/cell), making the PGM workload per leaf equivalent.

### E) CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--queries Q` | 100000 | Number of random point lookups |
| `--seed S` | 42 | RNG seed for query sampling |
| `--fanout F` | 10 | Grid divisions per dimension (F³ cells total) |
| `--max_depth D` | 1 | Max recursion depth for spatial partitioning |
| `--min_leaf M` | 100 | Stop splitting when cell has ≤ M points |
| `--split_thr T` | — | Alias for `--min_leaf` |
| `--model TYPE` | pgm | Leaf model: `pgm` \| `linear` \| `midpoint` |
| `--epsilon E` | 64 | PGM search window: `8` \| `16` \| `32` \| `64` \| `128` |

**Epsilon** controls the PGM refinement window (width ≈ 2ε+2). Smaller ε
means a tighter search window, lower PredError, and faster queries — at the
cost of a slightly larger PGM model per leaf. All values guarantee 100%
correctness on both Dense and Sparse 1M datasets.

### F) Epsilon sweep (Dense3D_1M, 100K queries, fanout=10)

```bash
for E in 8 16 32 64 128; do
  echo "--- epsilon=$E ---"
  build/bin/bench_rsmi3d_true3d \
    build/data/dense3d_1m_points.tpie \
    build/data/dense3d_1m_offsets.bin \
    1000000 Dense3D_1M \
    --queries 100000 --seed 42 --fanout 10 --epsilon $E
done
```

**Reference results** (Dense3D_1M, Q=100K, fanout=10, model=pgm):

| ε | PredError mean | PredError max | RefineWindow mean | Throughput | Correctness |
|---|---------------|--------------|-------------------|-----------|-------------|
| 8 | 3.81 | 9 | 17.9 | ~3.7M q/s | 100% |
| 16 | 7.60 | 17 | 33.7 | ~3.8M q/s | 100% |
| 32 | 15.24 | 33 | 64.9 | ~3.5M q/s | 100% |
| **64** | **29.94** | **65** | **125.5** | **~2.9M q/s** | **100%** ← default |
| 128 | 59.13 | 129 | 241.0 | ~3.4M q/s | 100% |

**Key observation**: PredError max ≈ ε+1 (PGM guarantee), RefineWindow ≈ 2ε+2.

---

## Experiment 4 — RSMI3D Scalability: N = 10K, 100K, 1M (Dense + Sparse)

Goal: measure how build time, index size, and query performance scale with
dataset size for both dense (Cartesian product grid) and sparse (random
uniform sample from [0,999]³) distributions.

### A) Dataset generation

**Dense3D_10K** (10×10×100 grid):
```bash
python3 tools/datasets/gen_3d_datasets.py \
    --type dense --N 10000 \
    --out_prefix build/data/dense3d_10k --verify
build/bin/raw_to_tpie build/data/dense3d_10k_points.raw \
    build/data/dense3d_10k_points.tpie
```

**Sparse3D_10K** (10K from [0,999]³, seed=42):
```bash
python3 tools/datasets/gen_3d_datasets.py \
    --type sparse --N 10000 --seed 42 \
    --out_prefix build/data/sparse3d_10k --verify
build/bin/raw_to_tpie build/data/sparse3d_10k_points.raw \
    build/data/sparse3d_10k_points.tpie
```

**Dense3D_100K** (50×50×40 grid):
```bash
python3 tools/datasets/gen_3d_datasets.py \
    --type dense --N 100000 \
    --out_prefix build/data/dense3d_100k --verify
build/bin/raw_to_tpie build/data/dense3d_100k_points.raw \
    build/data/dense3d_100k_points.tpie
```

**Sparse3D_100K** (100K from [0,999]³, seed=42):
```bash
python3 tools/datasets/gen_3d_datasets.py \
    --type sparse --N 100000 --seed 42 \
    --out_prefix build/data/sparse3d_100k --verify
build/bin/raw_to_tpie build/data/sparse3d_100k_points.raw \
    build/data/sparse3d_100k_points.tpie
```

Dense3D_1M and Sparse3D_1M were generated in Experiment 3 (see above).

### B) Run the full scalability suite

```bash
python3 tools/experiments/run_rsmi3d_scalability.py \
    --epsilon 8 --fanout 10 --depth 1 --min_leaf 100 --model pgm \
    --queries 100000 --seed 42
```

Results are written to: `build/results/rsmi3d_scalability.csv`

Optional flags: `--out_csv <path>`, `--dry_run` (print commands only).

### C) Scalability results (ε=8, fanout=10, Q=100K, model=pgm)

| Dataset | N | Dist | Build (ms) | Index (MB) | Lat mean (ns) | Lat p99 (ns) | Throughput | PE max | RW mean | Correct |
|---------|---|------|-----------|------------|--------------|-------------|-----------|--------|---------|---------|
| Dense3D_10K | 10K | dense | 1 | 0.5 | 98 | 135 | ~10.2M q/s | 7 | 9.1 | 100% |
| Sparse3D_10K | 10K | sparse | 1 | 0.5 | 76 | 114 | ~13.1M q/s | 9 | 9.1 | 100% |
| Dense3D_100K | 100K | dense | 8 | 1.9 | 123 | 333 | ~8.1M q/s | 9 | 17.1 | 100% |
| Sparse3D_100K | 100K | sparse | 10 | 1.9 | 119 | 302 | ~8.4M q/s | 9 | 17.1 | 100% |
| Dense3D_1M | 1M | dense | 98 | 15.9 | 246 | 645 | ~4.1M q/s | 9 | 17.9 | 100% |
| Sparse3D_1M | 1M | sparse | 111 | 15.9 | 243 | 583 | ~4.1M q/s | 9 | 17.9 | 100% |

All runs: **100% correctness**, ε=8, 1000/1000 non-empty cells.

### D) Key observations

1. **Build time scales linearly with N**: 1 ms → 8–10 ms → 98–111 ms.
   ~10× per decade as expected for O(N log N) sort plus O(N/cells) PGM fitting.

2. **Index size scales linearly with N**: ~0.5 MB at 10K, ~1.9 MB at 100K,
   ~15.9 MB at 1M. The dominant cost is per-cell PGM storage proportional
   to points per leaf.

3. **Query latency increases with N**: 76–98 ns → 119–123 ns → 243–246 ns
   (mean). The PGM search window (≈ 2ε+2 ≈ 18 entries) does not grow with N,
   but larger cells mean a wider binary search range and more cache misses.

4. **Dense vs Sparse converge at 1M**: At 10K, sparse is ~22% faster (76 vs
   98 ns mean); at 100K, ~3% faster (119 vs 123 ns); at 1M, effectively
   identical (243 vs 246 ns). As N grows, both distributions fill all 1000
   cells equally (~1000 pts/cell), making PGM workload per leaf equivalent.

5. **PE max behaviour at 10K**: Dense 10K shows PE max=7 (below ε+1=9) while
   Sparse 10K hits 9. Dense 10K has only ~10 pts/cell arranged in a perfectly
   regular stride, which PGM fits almost exactly. Sparse 10K points are random
   samples from [0,999]³, giving less regular Morton keys and thus the full
   ε+1 window is exercised. For N≥100K, PE max=9=ε+1 for both distributions.

6. **PredError max = ε+1 = 9** for all N≥100K, confirming the PGM guarantee
   holds regardless of dataset size or distribution.

### E) Recommended plots

- **Build time vs N** (log-log): dense line + sparse line → verify linear scaling
- **Latency mean + p99 vs N** (log-log): dense + sparse → show cache-miss regime
- **Index size vs N** (log-log): both lines should be parallel
- **Throughput vs N**: inverse latency trend

CSV columns available for all plots: `build/results/rsmi3d_scalability.csv`
