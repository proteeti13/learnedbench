// indexes/rsmi3d/rsmi3d.hpp
//
// RSMI-style 3D Learned Index  —  no libtorch required.
//
// Architecture
// ─────────────
// Recursive spatial partitioning: the 3D coordinate space is divided into a
// uniform fanout×fanout×fanout grid at each level.  Recursion continues until
// depth == max_depth  OR  a node contains ≤ min_leaf_size points.
//
// Each leaf stores its points sorted by global 3D Morton code and fits one of
// three lightweight predictors:
//   PGM      – PGM-Index on (morton_code → rank)            [default, most accurate]
//   LINEAR   – linear interpolation from m_first→m_last     [fast, medium accuracy]
//   MIDPOINT – constant prediction at leaf midpoint          [simplest, full-window]
//
// Point-lookup:
//   1. Traverse tree: at each internal node, compute child cell for (x,y,z).
//   2. At leaf: encode (x,y,z) as Morton code, apply model → (pred, lo, hi).
//   3. Binary-search [lo, hi) for exact Morton code.
//   4. Return aligned offset on hit; {false,0} on miss.
//
// Backward-compatible with bench_rsmi3d.cpp:
//   RSMI3D<64> idx(fanout);   ← still works (depth=1, pgm, min_leaf=100)
//
// Requires -mbmi2 compile flag.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "../pgm/pgm_index.hpp"
#include "../pgm/morton_nd.hpp"

namespace bench { namespace index {

// ---------------------------------------------------------------------------
// Model type
// ---------------------------------------------------------------------------
enum class ModelType { PGM, LINEAR, MIDPOINT };

inline std::string model_type_str(ModelType m) {
    if (m == ModelType::LINEAR)   return "linear";
    if (m == ModelType::MIDPOINT) return "midpoint";
    return "pgm";
}

// ---------------------------------------------------------------------------
// Per-query statistics
// ---------------------------------------------------------------------------
struct RSMI3DQueryStats {
    uint64_t pred_error  = 0;  ///< |pred_rank – true_rank|
    uint64_t corr_steps  = 0;  ///< binary-search steps in [lo, hi)
    uint64_t window_size = 0;  ///< refinement window width = hi − lo
    uint64_t latency_ns  = 0;  ///< wall-clock time for this lookup (ns)
    bool     found       = false;
};

// ---------------------------------------------------------------------------
// RSMI3D
// ---------------------------------------------------------------------------

template<size_t Epsilon = 64>
class RSMI3D {

    using Morton3D = mortonnd::MortonNDBmi<3, uint64_t>;
    static constexpr uint64_t MORTON_BITS = Morton3D::FieldBits;   // 21
    static constexpr uint64_t MORTON_MAX  = (1ULL << MORTON_BITS) - 1;

    // ------------------------------------------------------------------ leaf

    struct LeafData {
        std::vector<uint64_t> morton_codes; ///< sorted ascending
        std::vector<uint64_t> offsets;      ///< parallel to morton_codes
        ModelType model_type = ModelType::PGM;

        // PGM model
        pgm::PGMIndex<uint64_t, Epsilon> pgm;

        // LINEAR model: rank ≈ round(t * (n−1))  where t = (m−m0)/(m1−m0)
        uint64_t lin_m_first = 0, lin_m_last = 0;
        size_t   lin_n       = 0;

        bool built = false;

        void build(ModelType mt) {
            model_type = mt;
            lin_n      = morton_codes.size();
            if (lin_n == 0) return;
            lin_m_first = morton_codes.front();
            lin_m_last  = morton_codes.back();
            if (mt == ModelType::PGM) {
                pgm = pgm::PGMIndex<uint64_t, Epsilon>(
                          morton_codes.begin(), morton_codes.end());
            }
            built = true;
        }

        /// Returns {pred_pos, lo, hi} clamped to [0, n).
        std::array<size_t, 3> predict(uint64_t mc) const {
            const size_t n = morton_codes.size();
            if (n == 0) return {0, 0, 0};

            if (model_type == ModelType::PGM && built) {
                auto r = pgm.search(mc);
                return { r.pos,
                         std::min(r.lo, n),
                         std::min(r.hi, n) };
            }
            if (model_type == ModelType::LINEAR && lin_n > 1 &&
                lin_m_last > lin_m_first) {
                double t = static_cast<double>(mc - lin_m_first) /
                           static_cast<double>(lin_m_last - lin_m_first);
                t = std::max(0.0, std::min(1.0, t));
                size_t pred = static_cast<size_t>(t * (lin_n - 1));
                size_t lo   = pred >= Epsilon ? pred - Epsilon : 0;
                size_t hi   = std::min(pred + Epsilon + 2, n);
                return {pred, lo, hi};
            }
            // MIDPOINT or fallback
            return {n / 2, 0, n};
        }
    };

    // --------------------------------------------------------------- pool node
    // We use a flat pool of PoolNodes (no recursive type).  Children of node i
    // are stored at pool indices [node.children_base, node.children_base + fanout^3).

    struct PoolNode {
        bool is_leaf     = true;
        // Bounding box used during build and lookup traversal
        double min_x{}, max_x{}, min_y{}, max_y{}, min_z{}, max_z{};
        double range_x{}, range_y{}, range_z{};
        double cell_w_x{}, cell_w_y{}, cell_w_z{};   // sub-cell width at this level
        // Leaf: index into leaf_pool
        size_t leaf_idx       = 0;
        // Internal: first child index in node_pool
        size_t children_base  = 0;
    };

    // ------------------------------------------------------------------ fields
    std::vector<PoolNode> _node_pool;
    std::vector<LeafData> _leaf_pool;

    size_t    _fanout;
    size_t    _max_depth;
    size_t    _min_leaf_size;
    ModelType _model_type;

    // Global normalization (for Morton encoding)
    double _g_min_x{}, _g_range_x{};
    double _g_min_y{}, _g_range_y{};
    double _g_min_z{}, _g_range_z{};

    size_t _n            = 0;
    size_t _build_time_ms = 0;
    size_t _num_leaves   = 0;   ///< non-empty leaf nodes

public:
    // ---------------------------------------------------------------- ctor

    explicit RSMI3D(size_t fanout       = 10,
                    size_t max_depth    = 1,
                    size_t min_leaf_size = 100,
                    ModelType model     = ModelType::PGM)
        : _fanout(fanout), _max_depth(max_depth),
          _min_leaf_size(min_leaf_size), _model_type(model)
    {}

    // -------------------------------------------------------------- build

    void build(const std::vector<std::array<double, 3>>& points,
               const std::vector<uint64_t>&               offsets) {
        assert(points.size() == offsets.size());
        _n = points.size();
        _node_pool.clear();
        _leaf_pool.clear();
        _num_leaves = 0;

        auto t0 = std::chrono::steady_clock::now();

        // Global bounding box
        double min_x =  std::numeric_limits<double>::max();
        double max_x = -std::numeric_limits<double>::max();
        double min_y =  std::numeric_limits<double>::max();
        double max_y = -std::numeric_limits<double>::max();
        double min_z =  std::numeric_limits<double>::max();
        double max_z = -std::numeric_limits<double>::max();
        for (const auto& p : points) {
            if (p[0] < min_x) min_x = p[0]; if (p[0] > max_x) max_x = p[0];
            if (p[1] < min_y) min_y = p[1]; if (p[1] > max_y) max_y = p[1];
            if (p[2] < min_z) min_z = p[2]; if (p[2] > max_z) max_z = p[2];
        }
        _g_min_x = min_x; _g_range_x = max_x - min_x + 1.0;
        _g_min_y = min_y; _g_range_y = max_y - min_y + 1.0;
        _g_min_z = min_z; _g_range_z = max_z - min_z + 1.0;

        // Build root node (index 0)
        _node_pool.emplace_back();
        _node_pool[0].min_x   = min_x; _node_pool[0].max_x = max_x;
        _node_pool[0].min_y   = min_y; _node_pool[0].max_y = max_y;
        _node_pool[0].min_z   = min_z; _node_pool[0].max_z = max_z;
        _node_pool[0].range_x = _g_range_x;
        _node_pool[0].range_y = _g_range_y;
        _node_pool[0].range_z = _g_range_z;
        _node_pool[0].cell_w_x = _g_range_x / _fanout;
        _node_pool[0].cell_w_y = _g_range_y / _fanout;
        _node_pool[0].cell_w_z = _g_range_z / _fanout;

        // Build index pairs (point index only; we retrieve data by index)
        std::vector<size_t> idx_vec(points.size());
        std::iota(idx_vec.begin(), idx_vec.end(), 0);

        build_node(0, points, offsets, idx_vec, 0);

        auto t1 = std::chrono::steady_clock::now();
        _build_time_ms = static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count());
    }

    // -------------------------------------------------------------- lookup

    std::pair<bool, uint64_t>
    lookup(double x, double y, double z, RSMI3DQueryStats& stats) const {
        auto t0 = std::chrono::steady_clock::now();
        auto [found, offset] = lookup_node(0, x, y, z, stats);
        auto t1 = std::chrono::steady_clock::now();
        stats.latency_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
        stats.found = found;
        return {found, offset};
    }

    // ----------------------------------------------------------- accessors

    size_t build_time_ms()      const { return _build_time_ms; }
    size_t num_nonempty_cells() const { return _num_leaves; }
    size_t total_cells()        const { return _leaf_pool.size(); }
    size_t fanout()             const { return _fanout; }
    size_t max_depth()          const { return _max_depth; }
    size_t min_leaf_size()      const { return _min_leaf_size; }
    ModelType model_type()      const { return _model_type; }
    size_t n()                  const { return _n; }

    size_t size_in_bytes() const {
        size_t total = sizeof(*this)
                     + _node_pool.size() * sizeof(PoolNode)
                     + _leaf_pool.size() * sizeof(LeafData);
        for (const auto& lf : _leaf_pool) {
            total += lf.morton_codes.capacity() * sizeof(uint64_t);
            total += lf.offsets.capacity()      * sizeof(uint64_t);
            total += lf.pgm.size_in_bytes();
        }
        return total;
    }

private:
    // ------------------------------------------------------- build_node

    void build_node(size_t            node_idx,
                    const std::vector<std::array<double,3>>& pts,
                    const std::vector<uint64_t>&              offs,
                    const std::vector<size_t>&                indices,
                    size_t            depth)
    {
        const size_t cnt = indices.size();

        // ---- Leaf condition ----
        if (depth >= _max_depth || cnt <= _min_leaf_size) {
            make_leaf(node_idx, pts, offs, indices);
            return;
        }

        // ---- Internal node: partition into fanout^3 children ----
        // Read parent bounds into locals BEFORE any pool modification
        const double p_min_x   = _node_pool[node_idx].min_x;
        const double p_min_y   = _node_pool[node_idx].min_y;
        const double p_min_z   = _node_pool[node_idx].min_z;
        const double p_cell_wx = _node_pool[node_idx].cell_w_x;
        const double p_cell_wy = _node_pool[node_idx].cell_w_y;
        const double p_cell_wz = _node_pool[node_idx].cell_w_z;

        const size_t F   = _fanout;
        const size_t F2  = F * F;
        const size_t F3  = F * F * F;

        // Build per-cell buckets
        std::vector<std::vector<size_t>> buckets(F3);
        for (size_t pi : indices) {
            const auto& p = pts[pi];
            size_t cx = std::min(static_cast<size_t>((p[0]-p_min_x)/p_cell_wx), F-1);
            size_t cy = std::min(static_cast<size_t>((p[1]-p_min_y)/p_cell_wy), F-1);
            size_t cz = std::min(static_cast<size_t>((p[2]-p_min_z)/p_cell_wz), F-1);
            buckets[cx*F2 + cy*F + cz].push_back(pi);
        }

        // Allocate F3 children in one shot (avoids mid-loop reallocations)
        const size_t children_base = _node_pool.size();
        _node_pool[node_idx].is_leaf        = false;
        _node_pool[node_idx].children_base  = children_base;
        _node_pool.resize(children_base + F3);  // values preserved by resize

        // Set every child's bounding box (do this BEFORE recursing)
        for (size_t ci = 0; ci < F3; ++ci) {
            const size_t cx = ci / F2;
            const size_t cy = (ci / F) % F;
            const size_t cz = ci % F;
            const double cw_x = p_cell_wx / F;
            const double cw_y = p_cell_wy / F;
            const double cw_z = p_cell_wz / F;

            PoolNode& child       = _node_pool[children_base + ci];
            child.min_x   = p_min_x + cx * p_cell_wx;
            child.max_x   = child.min_x + p_cell_wx;
            child.min_y   = p_min_y + cy * p_cell_wy;
            child.max_y   = child.min_y + p_cell_wy;
            child.min_z   = p_min_z + cz * p_cell_wz;
            child.max_z   = child.min_z + p_cell_wz;
            child.range_x = p_cell_wx;
            child.range_y = p_cell_wy;
            child.range_z = p_cell_wz;
            child.cell_w_x = cw_x;
            child.cell_w_y = cw_y;
            child.cell_w_z = cw_z;
        }

        // Recurse into children
        for (size_t ci = 0; ci < F3; ++ci) {
            build_node(children_base + ci, pts, offs, buckets[ci], depth + 1);
        }
    }

    // ------------------------------------------------------- make_leaf

    void make_leaf(size_t            node_idx,
                   const std::vector<std::array<double,3>>& pts,
                   const std::vector<uint64_t>&              offs,
                   const std::vector<size_t>&                indices)
    {
        const size_t leaf_idx = _leaf_pool.size();
        _leaf_pool.emplace_back();
        LeafData& lf = _leaf_pool.back();

        _node_pool[node_idx].is_leaf  = true;
        _node_pool[node_idx].leaf_idx = leaf_idx;

        if (indices.empty()) return;

        lf.morton_codes.reserve(indices.size());
        lf.offsets.reserve(indices.size());

        for (size_t pi : indices) {
            lf.morton_codes.push_back(encode_morton(pts[pi][0], pts[pi][1], pts[pi][2]));
            lf.offsets.push_back(offs[pi]);
        }

        // Sort by Morton code
        std::vector<size_t> ord(lf.morton_codes.size());
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
            return lf.morton_codes[a] < lf.morton_codes[b];
        });
        std::vector<uint64_t> sm(lf.morton_codes.size()), so(lf.offsets.size());
        for (size_t j = 0; j < ord.size(); ++j) {
            sm[j] = lf.morton_codes[ord[j]];
            so[j] = lf.offsets[ord[j]];
        }
        lf.morton_codes = std::move(sm);
        lf.offsets      = std::move(so);

        lf.build(_model_type);
        ++_num_leaves;
    }

    // ------------------------------------------------------- lookup_node

    std::pair<bool, uint64_t>
    lookup_node(size_t node_idx, double x, double y, double z,
                RSMI3DQueryStats& stats) const
    {
        const PoolNode& node = _node_pool[node_idx];

        if (!node.is_leaf) {
            // Traverse to the right child
            const size_t F  = _fanout;
            const size_t F2 = F * F;
            size_t cx = std::min(
                static_cast<size_t>((x - node.min_x) / node.cell_w_x), F-1);
            size_t cy = std::min(
                static_cast<size_t>((y - node.min_y) / node.cell_w_y), F-1);
            size_t cz = std::min(
                static_cast<size_t>((z - node.min_z) / node.cell_w_z), F-1);
            return lookup_node(node.children_base + cx*F2 + cy*F + cz,
                               x, y, z, stats);
        }

        // ---- Leaf ----
        const LeafData& lf = _leaf_pool[node.leaf_idx];
        if (!lf.built) {
            stats.pred_error = stats.corr_steps = stats.window_size = 0;
            stats.found = false;
            return {false, 0};
        }

        const uint64_t mc = encode_morton(x, y, z);
        const auto [pred_pos, lo, hi] = lf.predict(mc);
        stats.window_size = hi - lo;

        size_t steps = 0;
        const auto it = lower_bound_counted(
            lf.morton_codes.begin() + lo,
            lf.morton_codes.begin() + hi,
            mc, steps);

        const bool found = (it != lf.morton_codes.begin() + hi) && (*it == mc);
        const size_t true_pos = static_cast<size_t>(
            std::distance(lf.morton_codes.begin(), it));

        stats.pred_error = (pred_pos >= true_pos)
                         ? (pred_pos - true_pos) : (true_pos - pred_pos);
        stats.corr_steps = steps;

        return {found, found ? lf.offsets[true_pos] : 0};
    }

    // ------------------------------------------------------- helpers

    inline uint64_t encode_morton(double x, double y, double z) const {
        const uint64_t ix = std::min(
            static_cast<uint64_t>((MORTON_MAX+1) * (x - _g_min_x) / _g_range_x),
            MORTON_MAX);
        const uint64_t iy = std::min(
            static_cast<uint64_t>((MORTON_MAX+1) * (y - _g_min_y) / _g_range_y),
            MORTON_MAX);
        const uint64_t iz = std::min(
            static_cast<uint64_t>((MORTON_MAX+1) * (z - _g_min_z) / _g_range_z),
            MORTON_MAX);
        return Morton3D::Encode(ix, iy, iz);
    }

    template<typename Iter, typename Val>
    static Iter lower_bound_counted(Iter first, Iter last,
                                    const Val& val, size_t& steps) {
        steps = 0;
        size_t count = static_cast<size_t>(std::distance(first, last));
        while (count > 0) {
            ++steps;
            const size_t half = count / 2;
            Iter mid = first;
            std::advance(mid, half);
            if (*mid < val) { first = std::next(mid); count -= half + 1; }
            else            { count = half; }
        }
        return first;
    }
};

}} // namespace bench::index
