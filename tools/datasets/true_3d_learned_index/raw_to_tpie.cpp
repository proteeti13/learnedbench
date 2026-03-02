// tools/datasets/true_3d_learned_index/raw_to_tpie.cpp
//
// Converts a raw binary doubles file (e.g. from gen_true3d_1m.py)
// into a TPIE file_stream<double> file for use with bench_rsmi3d.
//
// Usage:
//   raw_to_tpie <input.raw> <output.tpie>
//
// The input file must contain exactly N*D little-endian float64 values.
// The output is a valid tpie::file_stream<double> that can be opened with:
//   tpie::file_stream<double> fs; fs.open(output_path);

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

#include <tpie/tpie.h>
#include <tpie/file_stream.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.raw> <output.tpie>\n";
        return 1;
    }

    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];

    // Read raw doubles
    std::ifstream in(in_path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "Failed to open: " << in_path << "\n";
        return 1;
    }

    const std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size % sizeof(double) != 0) {
        std::cerr << "File size " << file_size
                  << " is not divisible by sizeof(double)=8\n";
        return 1;
    }

    const size_t n_doubles = static_cast<size_t>(file_size) / sizeof(double);
    std::cout << "Input : " << in_path << "  (" << n_doubles << " doubles, "
              << file_size / 1024 / 1024 << " MB)\n";

    std::vector<double> buf(n_doubles);
    if (!in.read(reinterpret_cast<char*>(buf.data()),
                 static_cast<std::streamsize>(file_size))) {
        std::cerr << "Read failed\n";
        return 1;
    }
    in.close();

    // Write TPIE file_stream<double>
    tpie::tpie_init();

    tpie::file_stream<double> out;
    out.open(out_path, tpie::access_write);

    for (double v : buf) {
        out.write(v);
    }

    out.close();
    tpie::tpie_finish();

    std::cout << "Output: " << out_path << "\n";
    std::cout << "Done. Converted " << n_doubles << " doubles.\n";
    return 0;
}
