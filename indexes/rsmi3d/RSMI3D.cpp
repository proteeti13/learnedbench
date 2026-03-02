#include "RSMI3D.h"
#include "./SortTools.h"

#include <cmath>
#include <algorithm>
#include <fstream>

using namespace std;
using namespace rsmi3dutil;
using namespace rsmi3dent;


// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------
RSMI3D::RSMI3D() {}

RSMI3D::RSMI3D(int idx, int max_part)
    : index(idx), max_partition_num(max_part), level(0) {}

RSMI3D::RSMI3D(int idx, int lvl, int max_part)
    : index(idx), level(lvl), max_partition_num(max_part) {}


// ---------------------------------------------------------------------------
// build()
// ---------------------------------------------------------------------------
void RSMI3D::build(ExpRecorder& exp_recorder, vector<Point> points)
{
    const int page_size = Constants::PAGESIZE;

    // ------------------------------------------------------------------
    // LEAF LEVEL:  points.size() <= exp_recorder.N  (e.g. 10 000)
    //
    // Ordering strategy:
    //   1. Sort by x → assign x_i rank
    //   2. Sort by y → assign y_i rank
    //   3. Sort by z → assign z_i rank
    //   4. Compute 3-D Hilbert curve value via generic hilbert.H API
    //   5. Sort by curve_val
    //   6. Assign normalized index ∈ [0,1]
    //   7. Pack into LeafNodes of PAGESIZE points each
    //   8. Train Net(3, …) on (x, y, z) → index
    // ------------------------------------------------------------------
    if ((long long)points.size() <= (long long)exp_recorder.N)
    {
        this->model_path += "_" + to_string(level) + "_" + to_string(index);
        if (exp_recorder.depth < level) exp_recorder.depth = level;
        exp_recorder.last_level_model_num++;
        is_last = true;
        N = (long long)points.size();

        // Number of bits per dimension for the Hilbert curve.
        // side = smallest power of 2 >= N; bits = log2(side).
        long long bits = (long long)ceil(log((double)N) / log(2.0));
        if (bits < 1) bits = 1;

        // --- assign x_i ---
        sort(points.begin(), points.end(), sortX());
        for (long long i = 0; i < N; i++) {
            points[i].x_i = i;
            mbr.update(points[i].x, points[i].y, points[i].z);
        }
        // --- assign y_i ---
        sort(points.begin(), points.end(), sortY());
        for (long long i = 0; i < N; i++) points[i].y_i = i;
        // --- assign z_i ---
        sort(points.begin(), points.end(), sortZ());
        for (long long i = 0; i < N; i++) points[i].z_i = i;

        // --- 3-D Hilbert curve value ---
        // Uses generic compute_Hilbert_value(long long[], size_t, long long)
        // declared in hilbert.H, implemented in hilbert.cpp.
        for (long long i = 0; i < N; i++) {
            long long coords[3] = { points[i].x_i, points[i].y_i, points[i].z_i };
            points[i].curve_val = compute_Hilbert_value(coords, 3, bits);
        }

        sort(points.begin(), points.end(), sort_curve_val());
        width = (int)(N - 1);

        if (N == 1) {
            points[0].index = 0;
        } else {
            for (long long i = 0; i < N; i++)
                points[i].index = (float)i / (float)(N - 1);
        }

        // --- pack into LeafNodes ---
        leaf_node_num = (int)(N / page_size);
        for (int i = 0; i < leaf_node_num; i++) {
            LeafNode ln;
            auto bn = points.begin() + i * page_size;
            auto en = points.begin() + i * page_size + page_size;
            ln.add_points(vector<Point>(bn, en));
            leafnodes.push_back(ln);
        }
        if ((long long)points.size() > (long long)(page_size * leaf_node_num)) {
            LeafNode ln;
            ln.add_points(vector<Point>(
                points.begin() + page_size * leaf_node_num, points.end()));
            leafnodes.push_back(ln);
            leaf_node_num++;
        }
        exp_recorder.leaf_node_num += leaf_node_num;

        // --- build training data: 3 floats (x,y,z) per point ---
        net = std::make_shared<Net>(3, leaf_node_num / 2 + 2);
        vector<float> locations;
        vector<float> labels;
        locations.reserve(N * 3);
        labels.reserve(N);
        for (auto& p : points) {
            locations.push_back(p.x);
            locations.push_back(p.y);
            locations.push_back(p.z);
            labels.push_back(p.index);
        }

        std::ifstream fin(this->model_path);
        if (!fin) {
            net->train_model(locations, labels);
            torch::save(net, this->model_path);
        } else {
            torch::load(net, this->model_path);
        }
        net->get_parameters();

        // --- compute error bounds for the refinement window ---
        exp_recorder.non_leaf_node_num++;
        for (long long i = 0; i < N; i++) {
            int predicted_idx = (int)(net->predict(points[i]) * leaf_node_num);
            predicted_idx = max(0, min(predicted_idx, leaf_node_num - 1));
            int error = (int)(i / page_size) - predicted_idx;
            if (error > 0)  { if (error > max_error) max_error = error; }
            else            { if (error < min_error) min_error = error; }
        }
        exp_recorder.average_max_error += max_error;
        exp_recorder.average_min_error += min_error;
        if ((max_error - min_error) > (exp_recorder.max_error - exp_recorder.min_error)) {
            exp_recorder.max_error = max_error;
            exp_recorder.min_error = min_error;
        }
    }
    else
    {
        // ------------------------------------------------------------------
        // INTERNAL LEVEL:  points.size() > exp_recorder.N
        //
        // 3-D Z-order grid: bit_num partitions per dimension.
        // Total grid cells = bit_num^3, Z-value ∈ [0, bit_num^3 - 1].
        //
        // Partitioning:
        //   1. Sort by x, divide into bit_num x-strips
        //   2. Within each x-strip, sort by y, divide into bit_num y-strips
        //   3. Within each y-strip, sort by z, divide into bit_num z-cells
        //   4. Assign 3-D Morton code to the cell
        //   5. Train Net(3) to predict Morton-code position
        //   6. Recurse into non-empty children
        // ------------------------------------------------------------------
        is_last = false;
        N = (long long)points.size();

        int bit_num  = max_partition_num;              // cells per dimension
        // Number of bits needed to encode indices 0..bit_num-1
        int num_bits = (int)ceil(log((double)bit_num) / log(2.0));
        if (num_bits < 1) num_bits = 1;

        long long side = 1LL;
        for (int d = 0; d < 3; d++) side *= bit_num;  // bit_num^3
        width = (int)(side - 1);

        map<int, vector<Point>> points_map;
        vector<float> locations(N * 3);
        vector<float> labels(N);
        long long point_index = 0;

        // Sort globally by x first
        sort(points.begin(), points.end(), sortX());

        long long x_strip_size = (N + bit_num - 1) / bit_num;

        for (int i = 0; i < bit_num; i++) {
            long long x_bn = (long long)i * x_strip_size;
            long long x_en = min((long long)(i + 1) * x_strip_size, N);
            if (x_bn >= N) break;

            vector<Point> x_strip(points.begin() + x_bn, points.begin() + x_en);
            sort(x_strip.begin(), x_strip.end(), sortY());

            long long y_strip_size = ((long long)x_strip.size() + bit_num - 1) / bit_num;

            for (int j = 0; j < bit_num; j++) {
                long long y_bn = (long long)j * y_strip_size;
                long long y_en = min((long long)(j + 1) * y_strip_size,
                                     (long long)x_strip.size());
                if (y_bn >= (long long)x_strip.size()) break;

                vector<Point> y_strip(x_strip.begin() + y_bn, x_strip.begin() + y_en);
                sort(y_strip.begin(), y_strip.end(), sortZ());

                long long z_cell_size = ((long long)y_strip.size() + bit_num - 1) / bit_num;

                for (int k = 0; k < bit_num; k++) {
                    long long z_bn = (long long)k * z_cell_size;
                    long long z_en = min((long long)(k + 1) * z_cell_size,
                                        (long long)y_strip.size());
                    if (z_bn >= (long long)y_strip.size()) break;

                    int Z_value = (int)compute_Z_value_3d(i, j, k, num_bits);

                    for (long long idx = z_bn; idx < z_en; idx++) {
                        Point& p = y_strip[idx];
                        p.index = (float)Z_value / (float)width;
                        locations[point_index * 3]     = p.x;
                        locations[point_index * 3 + 1] = p.y;
                        locations[point_index * 3 + 2] = p.z;
                        labels[point_index]             = p.index;
                        point_index++;
                        mbr.update(p.x, p.y, p.z);
                    }

                    if (points_map.find(Z_value) == points_map.end())
                        points_map[Z_value] = vector<Point>();
                }
            }
        }

        // --- train internal-level model with possible retrain ---
        bool is_retrain;
        do {
            this->model_path += "_" + to_string(level) + "_" + to_string(index);
            net = std::make_shared<Net>(3);

            std::ifstream fin(this->model_path);
            if (!fin) {
                net->train_model(locations, labels);
                torch::save(net, this->model_path);
            } else {
                torch::load(net, this->model_path);
            }
            net->get_parameters();

            // Assign each point to the bucket predicted by the model
            for (auto& p : points) {
                int predicted_idx = (int)(net->predict(p) * width);
                predicted_idx = max(0, min(predicted_idx, width - 1));
                points_map[predicted_idx].push_back(p);
            }

            int non_empty = 0;
            for (auto& kv : points_map)
                if (!kv.second.empty()) non_empty++;

            if (non_empty < 2) {
                // All points fell in one bucket – clear and retrain
                for (auto& kv : points_map) kv.second.clear();
                is_retrain = true;
            } else {
                is_retrain = false;
            }
        } while (is_retrain);

        exp_recorder.non_leaf_node_num++;
        points.clear();
        points.shrink_to_fit();

        // --- recurse into non-empty children ---
        for (auto& kv : points_map) {
            if (kv.second.empty()) continue;
            RSMI3D partition(kv.first, level + 1, max_partition_num);
            partition.model_path = model_path;
            partition.build(exp_recorder, kv.second);
            kv.second.clear();
            kv.second.shrink_to_fit();
            children[kv.first] = partition;
        }
    }
}


// ---------------------------------------------------------------------------
// print_index_info()
// ---------------------------------------------------------------------------
void RSMI3D::print_index_info(ExpRecorder& exp_recorder)
{
    cout << "  max_error           : " << exp_recorder.max_error << endl;
    cout << "  min_error           : " << exp_recorder.min_error << endl;
    cout << "  last_level_model_num: " << exp_recorder.last_level_model_num << endl;
    cout << "  leaf_node_num       : " << exp_recorder.leaf_node_num << endl;
    cout << "  non_leaf_node_num   : " << exp_recorder.non_leaf_node_num << endl;
    cout << "  depth               : " << exp_recorder.depth << endl;
}


// ---------------------------------------------------------------------------
// point_query() – single query
//
// Identical "predict + bounded scan" pattern as 2-D RSMI, using 3-D
// Mbr::contains() and 3-D Point::operator==.
// ---------------------------------------------------------------------------
bool RSMI3D::point_query(ExpRecorder& exp_recorder, Point qp)
{
    if (is_last)
    {
        // Primary prediction
        int predicted_idx = (int)(net->predict(qp) * leaf_node_num);
        predicted_idx = max(0, min(predicted_idx, leaf_node_num - 1));

        LeafNode& ln = leafnodes[predicted_idx];
        if (ln.mbr.contains(qp)) {
            exp_recorder.page_access += 1;
            auto it = find(ln.children->begin(), ln.children->end(), qp);
            if (it != ln.children->end()) return true;
        }

        // Bounded scan with error window
        int front = max(0, predicted_idx + min_error);
        int back  = min(leaf_node_num - 1, predicted_idx + max_error);

        int gap = 1;
        int left  = predicted_idx - gap;
        int right = predicted_idx + gap;

        while (left >= front && right <= back) {
            // search left
            LeafNode& ll = leafnodes[left];
            if (ll.mbr.contains(qp)) {
                exp_recorder.page_access += 1;
                for (auto& p : *ll.children)
                    if (p == qp) return true;
            }
            // search right
            LeafNode& rl = leafnodes[right];
            if (rl.mbr.contains(qp)) {
                exp_recorder.page_access += 1;
                for (auto& p : *rl.children)
                    if (p == qp) return true;
            }
            gap++;
            left  = predicted_idx - gap;
            right = predicted_idx + gap;
        }
        while (left >= front) {
            LeafNode& ll = leafnodes[left];
            if (ll.mbr.contains(qp)) {
                exp_recorder.page_access += 1;
                for (auto& p : *ll.children)
                    if (p == qp) return true;
            }
            gap++;
            left = predicted_idx - gap;
        }
        while (right <= back) {
            LeafNode& rl = leafnodes[right];
            if (rl.mbr.contains(qp)) {
                exp_recorder.page_access += 1;
                for (auto& p : *rl.children)
                    if (p == qp) return true;
            }
            gap++;
            right = predicted_idx + gap;
        }
        return false;
    }
    else
    {
        // Internal node: follow predicted child
        int predicted_idx = (int)(net->predict(qp) * width);
        predicted_idx = max(0, min(predicted_idx, width - 1));
        if (children.count(predicted_idx) == 0) return false;
        return children[predicted_idx].point_query(exp_recorder, qp);
    }
}


// ---------------------------------------------------------------------------
// point_query() – batch
// ---------------------------------------------------------------------------
void RSMI3D::point_query(ExpRecorder& exp_recorder, vector<Point> query_points)
{
    long sz = query_points.size();
    for (long i = 0; i < sz; i++) {
        auto start  = chrono::high_resolution_clock::now();
        point_query(exp_recorder, query_points[i]);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    }
    exp_recorder.time        /= sz;
    exp_recorder.page_access  = exp_recorder.page_access / sz;
}
