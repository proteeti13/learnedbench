// RSMI3D.h – True 3-D Recursive Spatial-Model Index
// ===================================================
// Drop-in 3-D fork of the 2-D RSMI (indexes/rsmi/RSMI.h).
//
// Key algorithmic differences from the 2-D original:
//   Leaf level  : 3-D Hilbert curve via hilbert.H generic N-dim API
//   Internal    : 3-D Z-order (Morton) grid, bit_num^3 partitions
//   Neural model: Net(3, …) – 3-D input (x, y, z)
//   Mbr / Point : extended with z dimension
//
// Only build() and point_query() are needed for the thesis benchmark;
// window_query, kNN_query, insert, remove are stubbed.

#ifndef RSMI3D_H
#define RSMI3D_H

#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <memory>

#include "./Node.h"
#include "./Point.h"
#include "./Mbr.h"
#include "./NonLeafNode.h"
#include "./LeafNode.h"
#include "./hilbert.H"
#include "./hilbert4.H"
#include "./z3d.H"
#include "./ModelTools.h"
#include "./ExpRecorder.h"

#include <boost/smart_ptr/make_shared_object.hpp>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/optim.h>
#include <torch/types.h>
#include <torch/utils.h>

using namespace at;
using namespace torch::nn;
using namespace torch::optim;
using namespace std;
using namespace rsmi3dutil;
using namespace rsmi3dent;

class RSMI3D
{
private:
    int level;
    int index;
    int max_partition_num;
    long long N = 0;
    int max_error = 0;
    int min_error = 0;
    int width = 0;
    int leaf_node_num = 0;
    bool is_last;

    Mbr mbr;
    std::shared_ptr<Net> net;

public:
    string model_path;
    map<int, RSMI3D> children;
    vector<LeafNode> leafnodes;

    RSMI3D();
    RSMI3D(int index, int max_partition_num);
    RSMI3D(int index, int level, int max_partition_num);

    void build(ExpRecorder& exp_recorder, vector<Point> points);
    void print_index_info(ExpRecorder& exp_recorder);

    bool point_query(ExpRecorder& exp_recorder, Point query_point);
    void point_query(ExpRecorder& exp_recorder, vector<Point> query_points);
};

#endif
