// bench/bench_rsmi3d_true3d.cpp
//
// RSMI3D true-3D point-lookup benchmark.
// Runs on ONE dataset at a time; call twice for dense + sparse comparison.
//
// CLI:
//   bench_rsmi3d_true3d <points.tpie> <offsets.bin> <N> <dataset_label>
//       [--queries     Q   ]   default 100000
//       [--seed        S   ]   default 42
//       [--fanout      F   ]   default 10
//       [--max_depth   D   ]   default 1
//       [--min_leaf    M   ]   default 100
//       [--split_thr   T   ]   alias for --min_leaf (region split threshold)
//       [--model       TYPE]   pgm | linear | midpoint   (default pgm)
//       [--epsilon     E   ]   PGM search window: 8 | 16 | 32 | 64 | 128
//                              (default 64)
//
// Metrics printed (ZMI-compatible + extended):
//   Build time (s), index size, latency p50/p95/p99, throughput,
//   PredError mean/p95/p99, RefineWindow mean/p95/p99, Correctness.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <tpie/tpie.h>
#include <tpie/file_stream.h>

#include "../indexes/rsmi3d/rsmi3d.hpp"

using namespace bench::index;

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

static std::vector<std::array<double,3>>
read_points_tpie(const std::string& path, size_t N) {
    tpie::tpie_init();
    std::vector<std::array<double,3>> pts;
    pts.reserve(N);
    tpie::file_stream<double> fs;
    fs.open(path);
    for (size_t i = 0; i < N; ++i) {
        pts.push_back({fs.read(), fs.read(), fs.read()});
    }
    fs.close();
    tpie::tpie_finish();
    return pts;
}

static std::vector<uint64_t>
read_offsets_bin(const std::string& path, size_t N) {
    std::vector<uint64_t> offs(N);
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << "\n"; std::exit(1); }
    if (!f.read(reinterpret_cast<char*>(offs.data()),
                static_cast<std::streamsize>(N * sizeof(uint64_t)))) {
        std::cerr << "Short read: " << path << "\n"; std::exit(1);
    }
    return offs;
}

// ---------------------------------------------------------------------------
// Stats helpers
// ---------------------------------------------------------------------------

template<typename T>
static double mean_of(const std::vector<T>& v) {
    if (v.empty()) return 0.0;
    return static_cast<double>(
        std::accumulate(v.begin(), v.end(), uint64_t{0})) / v.size();
}

template<typename T>
static T pct(std::vector<T> v, double p) {         // takes a copy → sorts
    if (v.empty()) return T{0};
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(p / 100.0 * (v.size() - 1))];
}

// ---------------------------------------------------------------------------
// Templated benchmark core — one instantiation per supported epsilon.
// PGMIndex<uint64_t, Eps> is a compile-time type, so we template on Eps here
// and dispatch at runtime via a switch in main().
// ---------------------------------------------------------------------------

template<size_t Eps>
static int run_bench(
    const std::vector<std::array<double,3>>& points,
    const std::vector<uint64_t>&             offsets,
    const std::string& label,
    size_t N, size_t Q, size_t seed,
    size_t fanout, size_t max_depth, size_t min_leaf,
    ModelType model)
{
    // -----------------------------------------------------------------------
    // Header
    // -----------------------------------------------------------------------
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "  Dataset   : " << label                              << "\n";
    std::cout << "  Model     : RSMI3D\n";
    std::cout << "────────────────────────────────────────────────────────\n";
    std::cout << "  Parameters\n";
    std::cout << "    epsilon             : " << Eps                    << "\n";
    std::cout << "    fanout per node     : " << fanout                 << "\n";
    std::cout << "    max recursion depth : " << max_depth              << "\n";
    std::cout << "    min points per leaf : " << min_leaf               << "  (split threshold)\n";
    std::cout << "    model type per leaf : " << model_type_str(model)  << "\n";
    std::cout << "    queries             : " << Q                      << "\n";
    std::cout << "    seed                : " << seed                   << "\n";
    std::cout << "    N (dataset size)    : " << N                      << "\n";
    std::cout << "────────────────────────────────────────────────────────\n";

    // -----------------------------------------------------------------------
    // Build
    // -----------------------------------------------------------------------
    RSMI3D<Eps> idx(fanout, max_depth, min_leaf, model);
    idx.build(points, offsets);

    const double build_s = idx.build_time_ms() / 1000.0;
    std::cout << "  Build time          : " << std::fixed << std::setprecision(3)
              << build_s << " s  (" << idx.build_time_ms() << " ms)\n";
    std::cout << "  Index size          : " << idx.size_in_bytes()
              << " bytes  ("
              << std::fixed << std::setprecision(1)
              << idx.size_in_bytes() / 1024.0 / 1024.0 << " MB)\n";
    std::cout << "  Non-empty leaves    : " << idx.num_nonempty_cells()
              << " / " << idx.total_cells() << "\n";
    std::cout << "────────────────────────────────────────────────────────\n";

    // -----------------------------------------------------------------------
    // Sample Q queries from the loaded dataset (guaranteed to exist)
    // -----------------------------------------------------------------------
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, N - 1);
    std::vector<size_t> qidxs(Q);
    for (auto& qi : qidxs) qi = dist(rng);

    // -----------------------------------------------------------------------
    // Run queries
    // -----------------------------------------------------------------------
    std::vector<uint64_t> latencies(Q), pred_errs(Q), corr_steps(Q), windows(Q);
    size_t found_count = 0;

    for (size_t q = 0; q < Q; ++q) {
        const size_t qi = qidxs[q];
        RSMI3DQueryStats st;
        const auto [found, got] = idx.lookup(
            points[qi][0], points[qi][1], points[qi][2], st);

        latencies[q]  = st.latency_ns;
        pred_errs[q]  = st.pred_error;
        corr_steps[q] = st.corr_steps;
        windows[q]    = st.window_size;

        if (found && got == offsets[qi]) ++found_count;
    }

    // -----------------------------------------------------------------------
    // Aggregate
    // -----------------------------------------------------------------------
    const double lat_mean    = mean_of(latencies);
    const double pe_mean     = mean_of(pred_errs);
    const double win_mean    = mean_of(windows);
    const uint64_t total_ns  =
        std::accumulate(latencies.begin(), latencies.end(), uint64_t{0});
    const double throughput  = (Q * 1e9) / static_cast<double>(total_ns);

    // -----------------------------------------------------------------------
    // Print results
    // -----------------------------------------------------------------------
    std::cout << "  Metrics\n";
    std::cout << "    Latency  mean       : "
              << static_cast<uint64_t>(lat_mean) << " ns\n";
    std::cout << "    Latency  p50        : " << pct(latencies, 50) << " ns\n";
    std::cout << "    Latency  p95        : " << pct(latencies, 95) << " ns\n";
    std::cout << "    Latency  p99        : " << pct(latencies, 99) << " ns\n";
    std::cout << "    Throughput          : "
              << static_cast<uint64_t>(throughput) << " queries/sec\n";
    std::cout << "────────────────────────────────────────────────────────\n";
    std::cout << "    PredError mean      : "
              << std::fixed << std::setprecision(2) << pe_mean << "\n";
    std::cout << "    PredError p95       : " << pct(pred_errs, 95) << "\n";
    std::cout << "    PredError p99       : " << pct(pred_errs, 99) << "\n";
    std::cout << "    PredError max       : " << pct(pred_errs, 100) << "\n";
    std::cout << "────────────────────────────────────────────────────────\n";
    std::cout << "    RefineWindow mean   : "
              << std::fixed << std::setprecision(2) << win_mean << "\n";
    std::cout << "    RefineWindow p95    : " << pct(windows, 95) << "\n";
    std::cout << "    RefineWindow p99    : " << pct(windows, 99) << "\n";
    std::cout << "────────────────────────────────────────────────────────\n";
    // ZMI-compatible line
    std::cout << "    PredError(avg)=" << std::fixed << std::setprecision(4)
              << pe_mean
              << ", PredError(max)=" << pct(pred_errs, 100) << "\n";
    std::cout << "    CorrSteps(avg)=" << std::fixed << std::setprecision(4)
              << mean_of(corr_steps)
              << ", CorrSteps(max)=" << pct(corr_steps, 100) << "\n";
    std::cout << "    FallbackRate=N/A\n";
    std::cout << "────────────────────────────────────────────────────────\n";
    std::cout << "    Correctness         : " << found_count << " / " << Q
              << "  (" << std::fixed << std::setprecision(2)
              << (100.0 * found_count / Q) << "%)\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    return (found_count == Q) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Main — parse CLI, load data once, dispatch to the right run_bench<Eps>
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr
            << "Usage: " << argv[0]
            << " <points.tpie> <offsets.bin> <N> <dataset_label>\n"
            << "       [--queries Q] [--seed S] [--fanout F]\n"
            << "       [--max_depth D] [--min_leaf M] [--split_thr T]\n"
            << "       [--model pgm|linear|midpoint] [--epsilon 8|16|32|64|128]\n";
        return 1;
    }

    const std::string pts_path    = argv[1];
    const std::string offs_path   = argv[2];
    const size_t      N           = std::stoull(argv[3]);
    const std::string label       = argv[4];

    size_t    Q          = 100000;
    size_t    seed       = 42;
    size_t    fanout     = 10;
    size_t    max_depth  = 1;
    size_t    min_leaf   = 100;
    size_t    epsilon    = 64;
    ModelType model      = ModelType::PGM;

    for (int i = 5; i < argc - 1; ++i) {
        std::string a = argv[i];
        if      (a == "--queries"  ) Q         = std::stoull(argv[++i]);
        else if (a == "--seed"     ) seed      = std::stoull(argv[++i]);
        else if (a == "--fanout"   ) fanout    = std::stoull(argv[++i]);
        else if (a == "--max_depth") max_depth = std::stoull(argv[++i]);
        else if (a == "--min_leaf" ) min_leaf  = std::stoull(argv[++i]);
        else if (a == "--split_thr") min_leaf  = std::stoull(argv[++i]);
        else if (a == "--epsilon"  ) epsilon   = std::stoull(argv[++i]);
        else if (a == "--model"    ) {
            std::string m = argv[++i];
            if      (m == "linear"  ) model = ModelType::LINEAR;
            else if (m == "midpoint") model = ModelType::MIDPOINT;
            else                      model = ModelType::PGM;
        }
    }

    // Load data once (shared across any epsilon instantiation)
    auto points  = read_points_tpie(pts_path, N);
    auto offsets = read_offsets_bin(offs_path, N);

    // Dispatch to the correct compile-time template instantiation
    switch (epsilon) {
        case   8: return run_bench<  8>(points, offsets, label, N, Q, seed,
                                        fanout, max_depth, min_leaf, model);
        case  16: return run_bench< 16>(points, offsets, label, N, Q, seed,
                                        fanout, max_depth, min_leaf, model);
        case  32: return run_bench< 32>(points, offsets, label, N, Q, seed,
                                        fanout, max_depth, min_leaf, model);
        case  64: return run_bench< 64>(points, offsets, label, N, Q, seed,
                                        fanout, max_depth, min_leaf, model);
        case 128: return run_bench<128>(points, offsets, label, N, Q, seed,
                                        fanout, max_depth, min_leaf, model);
        default:
            std::cerr << "Unsupported epsilon: " << epsilon
                      << ". Supported values: 8, 16, 32, 64, 128.\n";
            return 1;
    }
}
