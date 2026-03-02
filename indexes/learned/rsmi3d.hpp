// rsmi3d.hpp – Benchmark wrapper for RSMI3D (True 3-D Learned Index)
// ===================================================================
// Mirrors the interface of indexes/learned/rsmi.hpp (2-D wrapper) so
// bench_rsmi3d.cpp can follow the same structure as bench_rsmi.cpp.
//
// Key differences from rsmi.hpp:
//   • Uses RSMI3D instead of RSMI
//   • Input points are 3-D (x=SourceID, y=Hop1_ID, z=Hop2_ID)
//   • point_query_eval() takes a 3-D BenchPoint
//   • No static_assert on dimensionality (explicitly built for 3-D)

#pragma once

#include "../rsmi3d/RSMI3D.h"

#include <array>
#include <chrono>
#include <string>
#include <vector>

namespace bench { namespace index {

// BenchPoint3D matches what bench::utils::read_points<3>() produces:
//   std::array<double,3>   p = { SourceID_norm, Hop1_ID_norm, Hop2_ID_norm }
using BenchPoint3D = std::array<double, 3>;

struct PointQueryResult3D {
    bool      found;
    double    page_access;   // leaf pages probed (refinement cost)
    long long latency_ns;
};

class RSMI3DWrapper
{
public:
    // ------------------------------------------------------------------
    // Constructor – builds and trains the 3-D index.
    //
    // pts       : 3-D points loaded via bench::utils::read_points<3>()
    // model_dir : directory where trained model files are saved/loaded
    // n         : number of points (= pts.size())
    // ------------------------------------------------------------------
    RSMI3DWrapper(std::vector<BenchPoint3D>& pts,
                  const std::string& model_dir,
                  size_t n)
        : N(n)
    {
        // Convert bench points to rsmi3d internal Point3D
        std::vector<rsmi3dent::Point> rsmi_points;
        rsmi_points.reserve(n);
        for (auto& p : pts) {
            rsmi_points.emplace_back(
                static_cast<float>(p[0]),
                static_cast<float>(p[1]),
                static_cast<float>(p[2]));
        }

        _rsmi3d = new RSMI3D(0, rsmi3dutil::Constants::MAX_WIDTH);
        _rsmi3d->model_path = model_dir;

        _exp_recorder = new rsmi3dutil::ExpRecorder();
        _exp_recorder->N = (int)n;          // all points fit in one leaf level

        auto t0 = std::chrono::high_resolution_clock::now();
        _rsmi3d->build(*_exp_recorder, rsmi_points);
        auto t1 = std::chrono::high_resolution_clock::now();

        _build_ms = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / 1e6;
    }

    ~RSMI3DWrapper()
    {
        delete _rsmi3d;
        delete _exp_recorder;
    }

    // ------------------------------------------------------------------
    // point_query_eval() – single per-query timing + page-access count.
    // The query point must be a point that exists in the dataset.
    // ------------------------------------------------------------------
    PointQueryResult3D point_query_eval(const BenchPoint3D& pt)
    {
        rsmi3dent::Point qp(
            static_cast<float>(pt[0]),
            static_cast<float>(pt[1]),
            static_cast<float>(pt[2]));

        // Use a fresh recorder for per-query tracking
        rsmi3dutil::ExpRecorder local_rec;
        local_rec.N          = _exp_recorder->N;
        local_rec.page_access = 0.0;

        auto t0 = std::chrono::high_resolution_clock::now();
        bool found = _rsmi3d->point_query(local_rec, qp);
        auto t1 = std::chrono::high_resolution_clock::now();

        PointQueryResult3D result;
        result.found       = found;
        result.page_access = local_rec.page_access;
        result.latency_ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        return result;
    }

    double get_build_time_ms() const { return _build_ms; }

    void print_index_info() { _rsmi3d->print_index_info(*_exp_recorder); }

private:
    size_t                    N;
    RSMI3D*                   _rsmi3d;
    rsmi3dutil::ExpRecorder*  _exp_recorder;
    double                    _build_ms = 0.0;
};

}} // namespace bench::index
