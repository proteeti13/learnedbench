// bench/bench_rsmi3d.cpp
//
// RSMI3D point-lookup benchmark — apples-to-apples with ZMI (bench3d_toronto).
//
// Metric definitions match ZMI's zmindex.hpp + pgm_index_variants.hpp:
//   PredError  = |predicted_rank – true_rank|  within the cell's sorted array
//   CorrSteps  = binary-search steps inside the PGM [lo, hi) window
//   Latency    = wall-clock time per point lookup  (nanoseconds)
//   Correctness= fraction of queries that return the exact expected offset
//
// CLI:
//   bench_rsmi3d <points.tpie> <offsets.bin> <N>
//                [--queries Q]    (default 10000)
//                [--seed    S]    (default 42)
//                [--fanout  F]    (default 10)
//
// Compile-time knob: RSMI3D_EPSILON (default 64, matches ZMI's INDEX_ERROR_THRESHOLD)
//
// Build (after cmake):
//   make bench_rsmi3d -j$(nproc)

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

// ---------------------------------------------------------------------------
// Compile-time epsilon (matches ZMI INDEX_ERROR_THRESHOLD=64)
// ---------------------------------------------------------------------------
#ifndef RSMI3D_EPSILON
#define RSMI3D_EPSILON 64
#endif

using namespace bench::index;

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

static std::vector<std::array<double, 3>>
read_points_tpie(const std::string& path, size_t N) {
    tpie::tpie_init();
    std::vector<std::array<double, 3>> pts;
    pts.reserve(N);

    tpie::file_stream<double> fs;
    fs.open(path);
    for (size_t i = 0; i < N; ++i) {
        const double x = fs.read();
        const double y = fs.read();
        const double z = fs.read();
        pts.push_back({x, y, z});
    }
    fs.close();
    tpie::tpie_finish();
    return pts;
}

static std::vector<uint64_t>
read_offsets_bin(const std::string& path, size_t N) {
    std::vector<uint64_t> offs(N);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Failed to open offsets file: " << path << "\n";
        std::exit(1);
    }
    if (!f.read(reinterpret_cast<char*>(offs.data()),
                static_cast<std::streamsize>(N * sizeof(uint64_t)))) {
        std::cerr << "Short read on offsets file: " << path << "\n";
        std::exit(1);
    }
    return offs;
}

// ---------------------------------------------------------------------------
// Percentile helper (sorts a copy)
// ---------------------------------------------------------------------------

template<typename T>
static double pct(std::vector<T> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    return static_cast<double>(v[idx]);
}

template<typename T>
static T pct_raw(std::vector<T> v, double p) {
    if (v.empty()) return T{0};
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(p / 100.0 * (v.size() - 1))];
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <points.tpie> <offsets.bin> <N>"
                  << " [--queries Q] [--seed S] [--fanout F]\n";
        return 1;
    }

    const std::string points_path  = argv[1];
    const std::string offsets_path = argv[2];
    const size_t      N            = std::stoull(argv[3]);

    size_t Q      = 10000;
    size_t seed   = 42;
    size_t fanout = 10;

    for (int i = 4; i < argc - 1; ++i) {
        const std::string arg = argv[i];
        if      (arg == "--queries") Q      = std::stoull(argv[++i]);
        else if (arg == "--seed")    seed   = std::stoull(argv[++i]);
        else if (arg == "--fanout")  fanout = std::stoull(argv[++i]);
    }

    // -----------------------------------------------------------------------
    // Header
    // -----------------------------------------------------------------------
    std::cout << "====================================\n";
    std::cout << "Construct RSMI3D Epsilon=" << RSMI3D_EPSILON << "\n";
    std::cout << "Dataset     : " << points_path << "\n";
    std::cout << "N           : " << N << "\n";
    std::cout << "Fanout      : " << fanout
              << "  (" << fanout << "x" << fanout << "x" << fanout
              << " = " << fanout*fanout*fanout << " cells)\n";
    std::cout << "Queries     : " << Q << "  (seed=" << seed << ")\n";
    std::cout << "------------------------------------\n";

    // -----------------------------------------------------------------------
    // Load data
    // -----------------------------------------------------------------------
    std::cout << "Loading points from TPIE ...\n";
    auto points  = read_points_tpie(points_path, N);
    std::cout << "Loading offsets ...\n";
    auto offsets = read_offsets_bin(offsets_path, N);

    // -----------------------------------------------------------------------
    // Build index  (mirrors ZMI constructor output)
    // -----------------------------------------------------------------------
    RSMI3D<RSMI3D_EPSILON> idx(fanout);
    idx.build(points, offsets);

    std::cout << "Build Time: " << idx.build_time_ms() << " [ms]\n";
    std::cout << "Index Size: " << idx.size_in_bytes()  << " Bytes\n";
    std::cout << "Non-empty cells: "
              << idx.num_nonempty_cells() << " / " << idx.total_cells() << "\n";
    std::cout << "------------------------------------\n";

    // -----------------------------------------------------------------------
    // Sample Q random point queries from the dataset
    // -----------------------------------------------------------------------
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    std::vector<size_t> qidxs(Q);
    for (auto& qi : qidxs) qi = dist(rng);

    // -----------------------------------------------------------------------
    // Run queries
    // -----------------------------------------------------------------------
    std::vector<uint64_t> latencies(Q), pred_errors(Q), corr_steps(Q);
    size_t found_count = 0;

    for (size_t q = 0; q < Q; ++q) {
        const size_t   qi = qidxs[q];
        const double   x  = points[qi][0];
        const double   y  = points[qi][1];
        const double   z  = points[qi][2];
        const uint64_t expected = offsets[qi];

        RSMI3DQueryStats stats;
        const auto [found, got] = idx.lookup(x, y, z, stats);

        latencies[q]   = stats.latency_ns;
        pred_errors[q] = stats.pred_error;
        corr_steps[q]  = stats.corr_steps;

        if (found && got == expected) ++found_count;
    }

    // -----------------------------------------------------------------------
    // Aggregate statistics
    // -----------------------------------------------------------------------
    const double lat_mean = static_cast<double>(
        std::accumulate(latencies.begin(),   latencies.end(),   uint64_t{0})) / Q;
    const double pe_mean  = static_cast<double>(
        std::accumulate(pred_errors.begin(), pred_errors.end(), uint64_t{0})) / Q;
    const double cs_mean  = static_cast<double>(
        std::accumulate(corr_steps.begin(),  corr_steps.end(),  uint64_t{0})) / Q;

    // Sorted copies for percentiles
    auto lat_s = latencies;   std::sort(lat_s.begin(), lat_s.end());
    auto pe_s  = pred_errors; std::sort(pe_s.begin(),  pe_s.end());
    auto cs_s  = corr_steps;  std::sort(cs_s.begin(),  cs_s.end());

    auto p50 = [](const std::vector<uint64_t>& v) {
        return v[static_cast<size_t>(0.50 * (v.size() - 1))]; };
    auto p95 = [](const std::vector<uint64_t>& v) {
        return v[static_cast<size_t>(0.95 * (v.size() - 1))]; };
    auto p99 = [](const std::vector<uint64_t>& v) {
        return v[static_cast<size_t>(0.99 * (v.size() - 1))]; };

    const uint64_t total_lat_ns =
        std::accumulate(latencies.begin(), latencies.end(), uint64_t{0});
    const double   throughput   = (Q * 1e9) / static_cast<double>(total_lat_ns);

    // -----------------------------------------------------------------------
    // Print — ZMI-compatible section first
    // -----------------------------------------------------------------------
    std::cout << "PointLookupStats:\n";
    std::cout << "PredError(avg)=" << pe_mean
              << ", PredError(max)=" << pe_s.back() << "\n";
    std::cout << "CorrSteps(avg)=" << cs_mean
              << ", CorrSteps(max)=" << cs_s.back() << "\n";
    std::cout << "FallbackRate=N/A (no explicit fallback path)\n";
    std::cout << "------------------------------------\n";

    // Extended section with percentiles and latency
    std::cout << "ExtendedStats:\n";
    std::cout << "  Latency(mean)=" << static_cast<uint64_t>(lat_mean) << " ns"
              << "  Latency(p50)=" << p50(lat_s) << " ns"
              << "  Latency(p95)=" << p95(lat_s) << " ns"
              << "  Latency(p99)=" << p99(lat_s) << " ns\n";
    std::cout << "  PredError(p50)=" << p50(pe_s)
              << "  PredError(p95)=" << p95(pe_s)
              << "  PredError(p99)=" << p99(pe_s) << "\n";
    std::cout << "  CorrSteps(p95)=" << p95(cs_s)
              << "  CorrSteps(p99)=" << p99(cs_s) << "\n";
    std::cout << "  Correctness=" << found_count << "/" << Q
              << " (" << std::fixed << std::setprecision(2)
              << (100.0 * found_count / Q) << "%)\n";
    std::cout << "  Throughput=" << static_cast<uint64_t>(throughput)
              << " queries/sec\n";
    std::cout << "====================================\n";

    // Non-zero exit if correctness < 100%
    return (found_count == Q) ? 0 : 1;
}
