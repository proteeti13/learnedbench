/**
 * csv_to_tpie_lfb.cpp  –  LexFlattenBaseline Dataset Converter
 * ==============================================================
 * Reads a semicolon-delimited CSV with columns: SourceID;Hop1_ID;Hop2_ID;Offset
 * Builds an order-preserving 1D key K:
 *     BASE2 = max(Hop2_ID) + 1
 *     BASE1 = (max(Hop1_ID) + 1) * BASE2
 *     K = SourceID * BASE1 + Hop1_ID * BASE2 + Hop2_ID
 * Writes a TPIE file_stream<double> dataset compatible with
 * bench::utils::read_points<2>(...) – the layout bench_lfb expects:
 *     [x0, y0, x1, y1, ..., xN-1, yN-1]
 * where x = double(K), y = mode-dependent (see --y_mode).
 *
 * --y_mode=const       y = 0.0  (default; degenerate 2D → cleanest 1D signal)
 * --y_mode=hop2        y = double(Hop2_ID)
 * --y_mode=offset_mod  y = double(Offset % 128)
 *
 * LEXFLATTENBASELINE RATIONALE:
 * We encode the 3D key (SourceID, Hop1_ID, Hop2_ID) into a single scalar K
 * preserving lexicographic order, then treat (K, 0) as a degenerate 2D point
 * so that RSMI's 2D spatial partitioning can be reused without modification.
 * This is the BASELINE experiment. The True3DLearnedIndex (Path B) will
 * remove this collapse and partition the full 3D space, enabling true
 * multidimensional learned indexing.
 *
 * OUTPUT FORMAT:
 * - TPIE file_stream<double>; no manual header, tpie handles it.
 * - Layout: x0 y0 x1 y1 ... (2 doubles per point, interleaved)
 * - Compatible with bench::utils::read_points<2>(pts, fname, N)
 *
 * Usage:
 *   csv_to_tpie_lfb <input.csv> <output.tpie> [options]
 *
 * Options:
 *   --y_mode=const|hop2|offset_mod   (default: const)
 *   --base2=N                        override BASE2
 *   --base1=N                        override BASE1
 *   --verify                         read back and print first 5 points
 *   --no_header                      treat first row as data (no header line)
 */

#include <tpie/tpie.h>
#include <tpie/file_stream.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// --------------------------------------------------------------------------
// Row struct
// --------------------------------------------------------------------------
struct Row {
    int64_t src;
    int64_t hop1;
    int64_t hop2;
    int64_t offset;
};

// --------------------------------------------------------------------------
// Argument parsing helpers
// --------------------------------------------------------------------------
static std::string get_arg(int argc, char** argv, const std::string& prefix, const std::string& def = "") {
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.rfind(prefix, 0) == 0)
            return a.substr(prefix.size());
    }
    return def;
}

static bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.csv> <output.tpie>"
                     " [--y_mode=const|hop2|offset_mod]"
                     " [--base2=N] [--base1=N]"
                     " [--verify] [--no_header]\n";
        return 1;
    }

    const std::string csv_path  = argv[1];
    const std::string tpie_path = argv[2];
    const std::string y_mode    = get_arg(argc, argv, "--y_mode=", "const");
    const bool verify           = has_flag(argc, argv, "--verify");
    const bool no_header        = has_flag(argc, argv, "--no_header");
    const int64_t override_base2 = [&]() -> int64_t {
        auto s = get_arg(argc, argv, "--base2=", "");
        return s.empty() ? -1 : std::stoll(s);
    }();
    const int64_t override_base1 = [&]() -> int64_t {
        auto s = get_arg(argc, argv, "--base1=", "");
        return s.empty() ? -1 : std::stoll(s);
    }();

    if (y_mode != "const" && y_mode != "hop2" && y_mode != "offset_mod") {
        std::cerr << "Unknown --y_mode: " << y_mode << "\n";
        return 1;
    }

    std::cout << "====================================\n";
    std::cout << "LexFlattenBaseline CSV → TPIE Converter\n";
    std::cout << "Input CSV : " << csv_path  << "\n";
    std::cout << "Output    : " << tpie_path << "\n";
    std::cout << "y_mode    : " << y_mode    << "\n";
    std::cout << "====================================\n";

    // ------------------------------------------------------------------
    // 1. Read CSV
    // ------------------------------------------------------------------
    std::ifstream ifs(csv_path);
    if (!ifs) {
        std::cerr << "ERROR: cannot open " << csv_path << "\n";
        return 1;
    }

    std::vector<Row> rows;
    std::string line;
    bool first_line = true;

    while (std::getline(ifs, line)) {
        if (first_line && !no_header) {
            if (line.find("SourceID") == std::string::npos) {
                std::cerr << "WARN: header not recognized: " << line
                          << " – continuing anyway.\n";
            }
            first_line = false;
            continue;
        }
        first_line = false;

        if (line.empty()) continue;

        // strip trailing \r if present (Windows line endings)
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::istringstream ss(line);
        std::string tok;
        Row r{};
        int col = 0;
        while (std::getline(ss, tok, ';')) {
            switch (col++) {
                case 0: r.src    = std::stoll(tok); break;
                case 1: r.hop1   = std::stoll(tok); break;
                case 2: r.hop2   = std::stoll(tok); break;
                case 3: r.offset = std::stoll(tok); break;
            }
        }
        if (col < 4) {
            std::cerr << "WARN: skipping malformed line: " << line << "\n";
            continue;
        }
        rows.push_back(r);
    }
    ifs.close();

    if (rows.empty()) {
        std::cerr << "ERROR: no data rows parsed.\n";
        return 1;
    }
    std::cout << "Parsed " << rows.size() << " rows.\n";

    // ------------------------------------------------------------------
    // 2. Compute column stats
    // ------------------------------------------------------------------
    int64_t min_src = rows[0].src,  max_src = rows[0].src;
    int64_t min_h1  = rows[0].hop1, max_h1  = rows[0].hop1;
    int64_t min_h2  = rows[0].hop2, max_h2  = rows[0].hop2;
    int64_t min_off = rows[0].offset, max_off = rows[0].offset;

    for (auto& r : rows) {
        min_src = std::min(min_src, r.src);  max_src = std::max(max_src, r.src);
        min_h1  = std::min(min_h1,  r.hop1); max_h1  = std::max(max_h1,  r.hop1);
        min_h2  = std::min(min_h2,  r.hop2); max_h2  = std::max(max_h2,  r.hop2);
        min_off = std::min(min_off, r.offset); max_off = std::max(max_off, r.offset);
    }

    std::cout << "Column stats:\n"
              << "  SourceID : [" << min_src << ", " << max_src << "]\n"
              << "  Hop1_ID  : [" << min_h1  << ", " << max_h1  << "]\n"
              << "  Hop2_ID  : [" << min_h2  << ", " << max_h2  << "]\n"
              << "  Offset   : [" << min_off << ", " << max_off << "]\n";

    // ------------------------------------------------------------------
    // 3. Compute bases
    //    BASE2 = max(Hop2_ID) + 1  (number of distinct Hop2 values)
    //    BASE1 = (max(Hop1_ID) + 1) * BASE2
    //    K = SourceID*BASE1 + Hop1_ID*BASE2 + Hop2_ID
    //    Gives a unique, order-preserving scalar for every (src, h1, h2).
    // ------------------------------------------------------------------
    int64_t BASE2 = (override_base2 > 0) ? override_base2 : (max_h2 + 1);
    int64_t BASE1 = (override_base1 > 0) ? override_base1 : ((max_h1 + 1) * BASE2);

    std::cout << "Bases: BASE1=" << BASE1 << "  BASE2=" << BASE2 << "\n";
    std::cout << "K range: [0, " << (max_src * BASE1 + max_h1 * BASE2 + max_h2) << "]\n";

    // ------------------------------------------------------------------
    // 4. Sort by (SourceID, Hop1_ID, Hop2_ID) and verify monotonicity
    // ------------------------------------------------------------------
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.src  != b.src)  return a.src  < b.src;
        if (a.hop1 != b.hop1) return a.hop1 < b.hop1;
        return a.hop2 < b.hop2;
    });

    bool mono_ok = true;
    int64_t prev_K = rows[0].src * BASE1 + rows[0].hop1 * BASE2 + rows[0].hop2;

    for (size_t i = 1; i < rows.size(); ++i) {
        int64_t K = rows[i].src * BASE1 + rows[i].hop1 * BASE2 + rows[i].hop2;
        if (K < prev_K) {
            std::cerr << "WARN: K not non-decreasing at row " << i
                      << " (prev=" << prev_K << " cur=" << K << ")\n";
            mono_ok = false;
        }
        prev_K = K;

        if (rows[i].offset != rows[i - 1].offset + 1) {
            std::cerr << "WARN: Offset not sequential at row " << i
                      << " (got " << rows[i].offset
                      << ", expected " << (rows[i - 1].offset + 1) << ")\n";
            mono_ok = false;
        }
    }
    if (mono_ok)
        std::cout << "Monotonicity check: PASSED (K non-decreasing, Offset+1 sequential)\n";

    // ------------------------------------------------------------------
    // 5. Write TPIE file_stream<double>
    //    Layout: x0, y0, x1, y1, ... (2 doubles per point)
    //    Compatible with bench::utils::read_points<2>(pts, fname, N)
    // ------------------------------------------------------------------
    tpie::tpie_init();

    {
        tpie::file_stream<double> out;
        out.open(tpie_path, tpie::access_write);

        for (auto& r : rows) {
            double K_val = static_cast<double>(r.src * BASE1 + r.hop1 * BASE2 + r.hop2);
            double y_val = 0.0;
            if (y_mode == "hop2")            y_val = static_cast<double>(r.hop2);
            else if (y_mode == "offset_mod") y_val = static_cast<double>(r.offset % 128);
            // else const → y_val stays 0.0

            out.write(K_val);   // x (dimension 0)
            out.write(y_val);   // y (dimension 1)
        }

        out.close();
    }

    std::cout << "Wrote " << rows.size() << " 2-D points to " << tpie_path << "\n";

    // ------------------------------------------------------------------
    // 6. Verify: read back first 5 points
    // ------------------------------------------------------------------
    if (verify) {
        std::cout << "\nVerification – first 5 points read back:\n";
        tpie::file_stream<double> in;
        in.open(tpie_path, tpie::access_read);

        int to_print = std::min<int>(5, static_cast<int>(rows.size()));
        for (int i = 0; i < to_print; ++i) {
            double x = in.read();
            double y = in.read();
            std::cout << "  [" << i << "] x=" << x << "  y=" << y << "\n";
        }
        in.close();
        std::cout << "Verification: PASSED\n";
    }

    tpie::tpie_finish();

    std::cout << "\nDone. Output: " << tpie_path << "\n";
    return 0;
}
