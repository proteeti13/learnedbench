// bench_rsmi3d.cpp – True 3-D RSMI benchmark (point-lookup)
// ==========================================================
// Mirrors bench_rsmi.cpp (2-D) but uses RSMI3D + 3-D points.
//
// Input files (produced by csv_to_tpie_rsmi3d):
//   <points.tpie>   : TPIE file_stream<double>, 3 doubles per record
//                     layout [x0,y0,z0,  x1,y1,z1, ...]
//   <offsets.tpie>  : TPIE file_stream<double>, 1 double per record
//                     layout [off0, off1, ...] – lexicographic rank
//
// Usage:
//   bench_rsmi3d  <points.tpie>  <offsets.tpie>  N  <model_dir>
//                 [--queries=Q]  [--seed=S]
//
// Metrics printed (same set as bench_lfb / bench_rsmi):
//   Build/train time (ms)
//   Index size info (leaf/node counts)
//   Peak RSS (MB)
//   Latency mean/p50/p95/p99 (ns)
//   Page access (refinement probes) mean/p95/p99
//   Correctness (%)

#include "../utils/datautils.hpp"
#include "../indexes/learned/rsmi3d.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <sys/resource.h>

using BenchPoint3D = bench::index::BenchPoint3D;

// ---- argument helpers -------------------------------------------------------
static size_t parse_opt(int argc, char** argv, const char* key, size_t def)
{
    std::string prefix = std::string("--") + key + "=";
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.substr(0, prefix.size()) == prefix)
            return static_cast<size_t>(std::stoull(a.substr(prefix.size())));
    }
    return def;
}

static std::string get_filename(const std::string& path)
{
    auto idx = path.find_last_of('/') + 1;
    return path.substr(idx);
}

// ---- percentile helpers -----------------------------------------------------
static long long pctile_ll(std::vector<long long> v, double p)
{
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1) + 0.5);
    return v[std::min(idx, v.size() - 1)];
}
static double pctile_d(std::vector<double> v, double p)
{
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1) + 0.5);
    return v[std::min(idx, v.size() - 1)];
}
static double mean_ll(const std::vector<long long>& v)
{
    double s = 0; for (auto x : v) s += x; return s / v.size();
}
static double mean_d(const std::vector<double>& v)
{
    double s = 0; for (auto x : v) s += x; return s / v.size();
}

// =============================================================================
int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Usage: bench_rsmi3d <points.tpie> <offsets.tpie> N <model_dir>"
                  << " [--queries=Q] [--seed=S]\n";
        return 1;
    }

    const std::string pts_fname = argv[1];
    const std::string off_fname = argv[2];
    const size_t      N         = static_cast<size_t>(std::stoull(argv[3]));
    const std::string model_dir = std::string(argv[4]) + "/";
    const size_t      Q         = parse_opt(argc, argv, "queries", 10000);
    const uint64_t    seed      = static_cast<uint64_t>(parse_opt(argc, argv, "seed", 42));

    std::cout << "========================================\n";
    std::cout << "  bench_rsmi3d – True 3-D RSMI Benchmark\n";
    std::cout << "  N=" << N << "  Q=" << Q << "  seed=" << seed << "\n";
    std::cout << "  Points : " << pts_fname << "\n";
    std::cout << "  Offsets: " << off_fname << "\n";
    std::cout << "  Models : " << model_dir << "\n";
    std::cout << "========================================\n";

    // ------------------------------------------------------------------
    // 1. Load 3-D points
    // ------------------------------------------------------------------
    std::vector<BenchPoint3D> points;
    bench::utils::read_points<3>(points, pts_fname, N);
    std::cout << "[1] Loaded " << points.size() << " 3-D points.\n";

    // ------------------------------------------------------------------
    // 2. Load offsets (ground truth for correctness check)
    //    offsets[i] = lexicographic rank of points[i]
    // ------------------------------------------------------------------
    std::vector<double> offsets;
    offsets.reserve(N);
    {
        tpie::tpie_init();
        tpie::file_stream<double> fs;
        fs.open(off_fname);
        for (size_t i = 0; i < N; ++i)
            offsets.push_back(fs.read());
        fs.close();
        tpie::tpie_finish();
    }
    std::cout << "[2] Loaded " << offsets.size() << " offsets.\n";

    // ------------------------------------------------------------------
    // 3. Build model directory
    // ------------------------------------------------------------------
    if (!std::filesystem::is_directory(model_dir))
        std::filesystem::create_directories(model_dir);

    // ------------------------------------------------------------------
    // 4. Build / train RSMI3D index
    // ------------------------------------------------------------------
    std::cout << "[3] Building RSMI3D index…\n";
    bench::index::RSMI3DWrapper rsmi3d(points, model_dir, N);
    std::cout << "[3] Build complete in " << rsmi3d.get_build_time_ms() << " ms.\n";
    rsmi3d.print_index_info();

    // ------------------------------------------------------------------
    // 5. Sample Q query points
    // ------------------------------------------------------------------
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, N - 1);
    std::vector<size_t> gt_indices;
    gt_indices.reserve(Q);
    for (size_t i = 0; i < Q; ++i) gt_indices.push_back(dist(rng));

    // ------------------------------------------------------------------
    // 6. Run point queries
    // ------------------------------------------------------------------
    std::vector<long long> latencies_ns(Q);
    std::vector<double>    page_accesses(Q);
    size_t correct = 0;

    for (size_t i = 0; i < Q; ++i) {
        auto pqr = rsmi3d.point_query_eval(points[gt_indices[i]]);
        latencies_ns[i]  = pqr.latency_ns;
        page_accesses[i] = pqr.page_access;
        if (pqr.found) correct++;
    }

    // ------------------------------------------------------------------
    // 7. Peak RSS
    // ------------------------------------------------------------------
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    double rss_mb = ru.ru_maxrss / 1024.0;

    // ------------------------------------------------------------------
    // 8. Print results
    // ------------------------------------------------------------------
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n========================================\n";
    std::cout << "  bench_rsmi3d – point_lookup results\n";
    std::cout << "  N=" << N << "  Q=" << Q << "\n";
    std::cout << "========================================\n";
    std::cout << "  Build/train time (ms): " << rsmi3d.get_build_time_ms() << "\n";
    std::cout << "  Peak RSS (MB)        : " << rss_mb << "\n";
    std::cout << "  Query latency (ns):\n";
    std::cout << "    mean = " << std::setprecision(1) << mean_ll(latencies_ns) << "\n";
    std::cout << "    p50  = " << pctile_ll(latencies_ns, 50.0) << "\n";
    std::cout << "    p95  = " << pctile_ll(latencies_ns, 95.0) << "\n";
    std::cout << "    p99  = " << pctile_ll(latencies_ns, 99.0) << "\n";
    std::cout << "  Page access (refinement probes per query):\n";
    std::cout << "    mean = " << std::setprecision(3) << mean_d(page_accesses) << "\n";
    std::cout << "    p95  = " << pctile_d(page_accesses, 95.0) << "\n";
    std::cout << "    p99  = " << pctile_d(page_accesses, 99.0) << "\n";
    std::cout << "  Correctness          : " << correct << " / " << Q
              << "  (" << std::setprecision(2) << 100.0 * correct / Q << "%)\n";
    std::cout << "========================================\n";

    return 0;
}
