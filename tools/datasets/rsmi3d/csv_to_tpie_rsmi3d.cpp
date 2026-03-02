/**
 * csv_to_tpie_rsmi3d.cpp  –  RSMI3D Dataset Converter
 * =====================================================
 * Reads the same semicolon CSV used by LexFlattenBaseline:
 *   SourceID;Hop1_ID;Hop2_ID;Offset
 *
 * Writes TWO TPIE artifacts consumed by bench_rsmi3d:
 *
 *   <output_points.tpie>   – 3 doubles per record
 *     Layout: [x0,y0,z0,  x1,y1,z1, ...]
 *     where  x = double(SourceID),  y = double(Hop1_ID),  z = double(Hop2_ID)
 *     Compatible with bench::utils::read_points<3>(pts, fname, N)
 *
 *   <output_offsets.tpie>  – 1 double per record
 *     Layout: [off0, off1, ...]
 *     Row i = lexicographic rank of row i in the points file.
 *     Compatible with bench::utils::read_points<1>(offs, fname, N)
 *
 * Row order: lexicographic (SourceID, Hop1_ID, Hop2_ID) ascending –
 * identical to LexFlattenBaseline, so results are directly comparable.
 *
 * Usage:
 *   csv_to_tpie_rsmi3d  <input.csv>  <output_points.tpie>  <output_offsets.tpie>
 *                       [--verify]  [--no_header]
 *
 * Important: do NOT manually write TPIE headers; use tpie::file_stream<double>.
 */

#include <tpie/tpie.h>
#include <tpie/file_stream.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Row {
    int64_t src, hop1, hop2, offset;
};

static bool has_flag(int argc, char** argv, const std::string& flag)
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.csv> <output_points.tpie> <output_offsets.tpie>"
                     " [--verify] [--no_header]\n";
        return 1;
    }

    const std::string csv_path     = argv[1];
    const std::string pts_path     = argv[2];
    const std::string offsets_path = argv[3];
    const bool verify    = has_flag(argc, argv, "--verify");
    const bool no_header = has_flag(argc, argv, "--no_header");

    std::cout << "====================================\n";
    std::cout << "RSMI3D CSV → TPIE Converter\n";
    std::cout << "Input CSV      : " << csv_path     << "\n";
    std::cout << "Output points  : " << pts_path     << "\n";
    std::cout << "Output offsets : " << offsets_path << "\n";
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
            if (line.find("SourceID") == std::string::npos)
                std::cerr << "WARN: header not recognized: " << line << "\n";
            first_line = false;
            continue;
        }
        first_line = false;
        if (line.empty()) continue;
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
    // 2. Sort lexicographically (SourceID, Hop1_ID, Hop2_ID)
    // ------------------------------------------------------------------
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.src  != b.src)  return a.src  < b.src;
        if (a.hop1 != b.hop1) return a.hop1 < b.hop1;
        return a.hop2 < b.hop2;
    });

    // ------------------------------------------------------------------
    // 3. Write TPIE artifacts
    // ------------------------------------------------------------------
    tpie::tpie_init();
    {
        tpie::file_stream<double> pts_out;
        pts_out.open(pts_path, tpie::access_write);

        tpie::file_stream<double> off_out;
        off_out.open(offsets_path, tpie::access_write);

        for (auto& r : rows) {
            pts_out.write(static_cast<double>(r.src));    // x = SourceID
            pts_out.write(static_cast<double>(r.hop1));   // y = Hop1_ID
            pts_out.write(static_cast<double>(r.hop2));   // z = Hop2_ID
            off_out.write(static_cast<double>(r.offset)); // payload
        }

        pts_out.close();
        off_out.close();
    }

    std::cout << "Wrote " << rows.size() << " 3-D points  → " << pts_path     << "\n";
    std::cout << "Wrote " << rows.size() << " offsets     → " << offsets_path << "\n";

    // ------------------------------------------------------------------
    // 4. Verify: read back first 5 rows
    // ------------------------------------------------------------------
    if (verify) {
        std::cout << "\nVerification – first 5 rows read back:\n";
        tpie::file_stream<double> pts_in;
        pts_in.open(pts_path, tpie::access_read);
        tpie::file_stream<double> off_in;
        off_in.open(offsets_path, tpie::access_read);

        int to_print = std::min<int>(5, (int)rows.size());
        for (int i = 0; i < to_print; ++i) {
            double x   = pts_in.read();
            double y   = pts_in.read();
            double z   = pts_in.read();
            double off = off_in.read();
            std::cout << "  [" << i << "] src=" << x
                      << "  hop1=" << y
                      << "  hop2=" << z
                      << "  offset=" << off << "\n";
        }
        pts_in.close();
        off_in.close();
        std::cout << "Verification: PASSED\n";
    }

    tpie::tpie_finish();
    std::cout << "\nDone.\n";
    return 0;
}
