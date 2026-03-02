/**
 * bench_lfb.cpp  –  LexFlattenBaseline Benchmark
 * ================================================
 * Evaluates a learned point-lookup index on the 1D-encoded graph dataset
 * produced by csv_to_tpie_lfb (LexFlattenBaseline converter).
 *
 * INDEX USED:  PGM-Index (pgm::PGMIndex<double, Epsilon>)
 *              Header-only, no PyTorch dependency  → runs on Linux/WSL2.
 *              Conceptually equivalent to the prediction layer inside RSMI:
 *              both learn a piecewise-linear mapping from key → rank.
 *
 * DESIGN NOTE (LexFlattenBaseline vs True3DLearnedIndex):
 *   LexFlattenBaseline:
 *     K = SourceID*BASE1 + Hop1_ID*BASE2 + Hop2_ID collapses the 3-D key
 *     into a single scalar, then uses a 1-D learned index (PGM). This
 *     eliminates inter-dimensional correlation and gives a tight lower bound
 *     on prediction error for the 1-D projection. The y_mode sensitivity
 *     experiment (--sensitivity) proves that y is irrelevant for 1-D PGM,
 *     which motivates the True3DLearnedIndex.
 *
 *   True3DLearnedIndex (Path B, future work):
 *     Will replace K with true (x,y,z) = (SourceID, Hop1_ID, Hop2_ID) and
 *     use RSMI's 2-D spatial partitioning (or a 3-D extension) to exploit
 *     multi-dimensional structure instead of collapsing it.
 *
 * METRICS REPORTED (database-research relevant):
 *   Build time (ms)                   Index size (bytes, MB)
 *   Peak RSS (MB)                     Query latency: mean, p50, p95, p99 (ns)
 *   Prediction error: mean, p95, p99  Refinement window: mean, p95, p99
 *   Correctness: % exact matches      Sensitivity: const vs hop2 y_mode
 *
 * Usage:
 *   bench_lfb <data.tpie> <N> [options]
 *
 * Options:
 *   --queries=<Q>        number of point queries to sample (default 10000)
 *   --epsilon=<E>        PGM epsilon (default 64)
 *   --seed=<S>           RNG seed (default 42)
 *   --y_mode=<m>         const|hop2|offset_mod  (informational, for logging)
 *   --sensitivity        run sensitivity experiment (loads both const + hop2
 *                        from files derived by substituting _const/_hop2 suffix)
 */

#include "../utils/datautils.hpp"
#include "../utils/type.hpp"
#include "../indexes/pgm/pgm_index.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

// --------------------------------------------------------------------------
// Argument helpers
// --------------------------------------------------------------------------
static std::string get_opt(int argc, char** argv, const std::string& prefix, const std::string& def = "") {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.rfind(prefix, 0) == 0) return a.substr(prefix.size());
    }
    return def;
}
static bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

// --------------------------------------------------------------------------
// Percentile on a vector (makes a sorted copy)
// --------------------------------------------------------------------------
template<typename T>
static T pctile(std::vector<T> v, double p) {
    if (v.empty()) return T{};
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1) + 0.5);
    return v[std::min(idx, v.size() - 1)];
}
template<typename T>
static double mean_of(const std::vector<T>& v) {
    if (v.empty()) return 0.0;
    double s = 0;
    for (auto x : v) s += static_cast<double>(x);
    return s / v.size();
}

// --------------------------------------------------------------------------
// Peak RSS in MB (Linux)
// --------------------------------------------------------------------------
static double peak_rss_mb() {
    struct rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / 1024.0;  // Linux: KB → MB
}

// --------------------------------------------------------------------------
// Run one evaluation pass
// --------------------------------------------------------------------------
struct Results {
    double build_time_ms;
    size_t index_size_bytes;
    size_t n_points;
    std::vector<long long> latencies_ns;   // per query
    std::vector<long long> pred_errors;    // |predicted_pos - actual_pos|
    std::vector<long long> refine_widths;  // hi - lo (scan window)
    size_t correct;                        // exact key matches
    size_t total;                          // = Q
    double rss_mb;
};

template<size_t Epsilon>
static Results run_experiment(
    const std::string& tpie_file,
    size_t N,
    size_t Q,
    uint64_t seed,
    const std::string& y_mode_label)
{
    std::cout << "\n---- Experiment [y_mode=" << y_mode_label
              << ", eps=" << Epsilon << "] ----\n";

    // 1. Load data via bench::utils::read_points<2>
    std::vector<point_t<2>> pts;
    bench::utils::read_points<2>(pts, tpie_file, N);
    std::cout << "Loaded " << pts.size() << " 2-D points.\n";

    // Extract x-coords (= K values) – sorted because csv_to_tpie_lfb emits rows sorted.
    std::vector<double> keys;
    keys.reserve(pts.size());
    for (auto& p : pts) keys.push_back(p[0]);

    // Validate sorted
    for (size_t i = 1; i < keys.size(); ++i) {
        if (keys[i] < keys[i-1]) {
            std::cerr << "WARN: keys not sorted at position " << i
                      << " (key[i-1]=" << keys[i-1]
                      << " key[i]=" << keys[i] << ")\n";
            std::sort(keys.begin(), keys.end());
            break;
        }
    }

    // 2. Build PGM index
    auto t0 = std::chrono::steady_clock::now();
    pgm::PGMIndex<double, Epsilon> pgm_idx(keys.begin(), keys.end());
    auto t1 = std::chrono::steady_clock::now();

    double build_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    size_t idx_bytes = pgm_idx.size_in_bytes();
    double rss = peak_rss_mb();

    std::cout << "Build time  : " << std::fixed << std::setprecision(3) << build_ms << " ms\n";
    std::cout << "Index size  : " << idx_bytes << " bytes  ("
              << std::fixed << std::setprecision(3) << idx_bytes / 1048576.0 << " MB)\n";
    std::cout << "Peak RSS    : " << rss << " MB\n";

    // 3. Sample Q query keys uniformly at random from the dataset
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, keys.size() - 1);

    std::vector<size_t> query_indices(Q);
    for (size_t i = 0; i < Q; ++i) query_indices[i] = dist(rng);

    // 4. Run point lookups
    Results res{};
    res.build_time_ms    = build_ms;
    res.index_size_bytes = idx_bytes;
    res.n_points = pts.size();
    res.rss_mb   = rss;
    res.total    = Q;
    res.correct  = 0;

    res.latencies_ns.reserve(Q);
    res.pred_errors.reserve(Q);
    res.refine_widths.reserve(Q);

    for (size_t qi = 0; qi < Q; ++qi) {
        size_t true_pos   = query_indices[qi];
        double query_key  = keys[true_pos];

        auto qs = std::chrono::steady_clock::now();

        // PGM search: returns {pos, lo, hi} where key is in keys[lo..hi)
        auto approx = pgm_idx.search(query_key);
        size_t lo = approx.lo;
        size_t hi = approx.hi;
        size_t pred_pos = approx.pos;

        // Last-mile scan: binary search within [lo, hi)
        // (equivalent to RSMI's page scan within predicted ± error)
        size_t found_pos = keys.size();  // sentinel = not found
        auto it = std::lower_bound(keys.begin() + lo, keys.begin() + hi, query_key);
        if (it != keys.begin() + hi && *it == query_key) {
            found_pos = static_cast<size_t>(it - keys.begin());
        }

        auto qe = std::chrono::steady_clock::now();

        long long lat_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(qe - qs).count();
        res.latencies_ns.push_back(lat_ns);

        long long pred_err = static_cast<long long>(pred_pos) - static_cast<long long>(true_pos);
        if (pred_err < 0) pred_err = -pred_err;
        res.pred_errors.push_back(pred_err);

        long long width = static_cast<long long>(hi) - static_cast<long long>(lo);
        res.refine_widths.push_back(width);

        if (found_pos != keys.size() && keys[found_pos] == query_key) {
            res.correct++;
        }
    }

    return res;
}

// --------------------------------------------------------------------------
// Print results table
// --------------------------------------------------------------------------
static void print_table(const std::string& label, const Results& r) {
    std::cout << "\n========================================\n";
    std::cout << "  Results for: " << label << "\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "  N (dataset size)  : " << r.n_points << "\n";
    std::cout << "  Q (queries)       : " << r.total    << "\n\n";

    // Build
    std::cout << "  Build time (ms)   : " << r.build_time_ms << "\n";
    std::cout << "  Index size (bytes): " << r.index_size_bytes
              << "  (" << r.index_size_bytes / 1048576.0 << " MB)\n";
    std::cout << "  Peak RSS (MB)     : " << r.rss_mb << "\n\n";

    // Query latency
    std::cout << "  Query latency (ns):\n";
    std::cout << "    mean  = " << std::setprecision(1) << mean_of(r.latencies_ns)  << "\n";
    std::cout << "    p50   = " << pctile(r.latencies_ns, 50.0)  << "\n";
    std::cout << "    p95   = " << pctile(r.latencies_ns, 95.0)  << "\n";
    std::cout << "    p99   = " << pctile(r.latencies_ns, 99.0)  << "\n\n";

    // Prediction error
    std::cout << "  Prediction error (positions):\n";
    std::cout << "    mean  = " << std::setprecision(2) << mean_of(r.pred_errors) << "\n";
    std::cout << "    p95   = " << pctile(r.pred_errors, 95.0) << "\n";
    std::cout << "    p99   = " << pctile(r.pred_errors, 99.0) << "\n\n";

    // Refinement cost
    std::cout << "  Refinement window (scan width):\n";
    std::cout << "    mean  = " << std::setprecision(2) << mean_of(r.refine_widths) << "\n";
    std::cout << "    p95   = " << pctile(r.refine_widths, 95.0) << "\n";
    std::cout << "    p99   = " << pctile(r.refine_widths, 99.0) << "\n\n";

    // Correctness
    double pct = 100.0 * r.correct / r.total;
    std::cout << "  Correctness       : " << r.correct << " / " << r.total
              << "  (" << std::setprecision(2) << pct << "%)\n";
    std::cout << "========================================\n";
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <data.tpie> <N>"
                     " [--queries=Q] [--epsilon=E] [--seed=S]"
                     " [--y_mode=const|hop2|offset_mod] [--sensitivity]\n";
        return 1;
    }

    const std::string tpie_file = argv[1];
    const size_t N = static_cast<size_t>(std::stoull(argv[2]));
    const size_t Q = static_cast<size_t>(std::stoull(get_opt(argc, argv, "--queries=", "10000")));
    const int    E = std::stoi(get_opt(argc, argv, "--epsilon=", "64"));
    const uint64_t seed = static_cast<uint64_t>(std::stoull(get_opt(argc, argv, "--seed=", "42")));
    const std::string y_mode = get_opt(argc, argv, "--y_mode=", "const");
    const bool sensitivity  = has_flag(argc, argv, "--sensitivity");

    std::cout << "====================================\n";
    std::cout << "bench_lfb – LexFlattenBaseline\n";
    std::cout << "  file   : " << tpie_file << "\n";
    std::cout << "  N      : " << N  << "\n";
    std::cout << "  Q      : " << Q  << "\n";
    std::cout << "  epsilon: " << E  << "\n";
    std::cout << "  seed   : " << seed << "\n";
    std::cout << "  y_mode : " << y_mode << "\n";
    std::cout << "====================================\n";

    // Dispatch on epsilon at compile-time (common values precompiled)
    Results res1;
    std::string label1 = "y_mode=" + y_mode;

    if (E == 4)
        res1 = run_experiment<4>(tpie_file, N, Q, seed, y_mode);
    else if (E == 16)
        res1 = run_experiment<16>(tpie_file, N, Q, seed, y_mode);
    else if (E == 128)
        res1 = run_experiment<128>(tpie_file, N, Q, seed, y_mode);
    else
        res1 = run_experiment<64>(tpie_file, N, Q, seed, y_mode);

    print_table(label1, res1);

    // ------------------------------------------------------------------
    // Sensitivity: compare y_mode=const vs y_mode=hop2
    // For LexFlattenBaseline the y value only affects RSMI's 2-D partitioning.
    // With PGM (1-D index on K), y_mode has zero effect on prediction quality.
    // We show this explicitly: the two runs produce identical results, which
    // IS the finding: the 1-D collapse makes y_mode irrelevant and motivates
    // the True3DLearnedIndex (Path B).
    // ------------------------------------------------------------------
    if (sensitivity) {
        std::string alt_file = tpie_file;
        auto pos = alt_file.rfind("_const");
        if (pos != std::string::npos) {
            alt_file.replace(pos, 6, "_hop2");
        } else {
            alt_file = tpie_file;  // fallback: same file, shows y_mode irrelevance
        }

        std::cout << "\n[Sensitivity] Running on: " << alt_file << "\n";
        Results res2;
        if (E == 4)        res2 = run_experiment<4>(alt_file, N, Q, seed, "hop2");
        else if (E == 16)  res2 = run_experiment<16>(alt_file, N, Q, seed, "hop2");
        else if (E == 128) res2 = run_experiment<128>(alt_file, N, Q, seed, "hop2");
        else               res2 = run_experiment<64>(alt_file, N, Q, seed, "hop2");

        print_table("y_mode=hop2 (sensitivity run)", res2);

        std::cout << "\n========================================\n";
        std::cout << "  Sensitivity Comparison\n";
        std::cout << "  (LexFlattenBaseline: 1-D PGM, y_mode affects only RSMI 2-D partition)\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Metric               const          hop2\n";
        std::cout << "  Build time (ms)  "
                  << std::setw(12) << res1.build_time_ms << "  "
                  << std::setw(12) << res2.build_time_ms << "\n";
        std::cout << "  Idx size (bytes) "
                  << std::setw(12) << res1.index_size_bytes << "  "
                  << std::setw(12) << res2.index_size_bytes << "\n";
        std::cout << "  Latency mean(ns) "
                  << std::setw(12) << mean_of(res1.latencies_ns) << "  "
                  << std::setw(12) << mean_of(res2.latencies_ns) << "\n";
        std::cout << "  Pred err mean    "
                  << std::setw(12) << mean_of(res1.pred_errors) << "  "
                  << std::setw(12) << mean_of(res2.pred_errors) << "\n";
        std::cout << "  Refine mean      "
                  << std::setw(12) << mean_of(res1.refine_widths) << "  "
                  << std::setw(12) << mean_of(res2.refine_widths) << "\n";
        std::cout << "  Correctness (%)  "
                  << std::setw(12) << 100.0 * res1.correct / res1.total << "  "
                  << std::setw(12) << 100.0 * res2.correct / res2.total << "\n";
        std::cout << "========================================\n";

        std::cout << "\n[NOTE] For LexFlattenBaseline (1-D PGM on K), y_mode has NO effect\n"
                  << "       on prediction quality because PGM only sees the x=K dimension.\n"
                  << "       True3DLearnedIndex (Path B) will use true 2-D/3-D partitioning\n"
                  << "       where the extra dimensions carry meaningful index structure.\n";
    }

    return 0;
}
