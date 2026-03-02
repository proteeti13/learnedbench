#ifndef EXPRECORDER3D_H
#define EXPRECORDER3D_H

#include <vector>
#include <string>
#include "./Constants.h"
using namespace std;

namespace rsmi3dutil {

class ExpRecorder
{
public:
    long long leaf_node_num    = 0;
    long long non_leaf_node_num = 0;

    int max_error = 0;
    int min_error = 0;
    int depth = 0;
    int last_level_model_num = 0;

    long long average_max_error = 0;
    long long average_min_error = 0;

    // N = max points per leaf-level node (set externally to dataset size).
    // When points.size() <= N, the node is treated as leaf.
    int N = Constants::THRESHOLD;

    long long insert_num   = 0;
    long delete_num        = 0;
    long long rebuild_time = 0;
    int  rebuild_num       = 0;

    long   time        = 0;
    long   insert_time = 0;
    long   delete_time = 0;
    double page_access = 1.0;
    long   size        = 0;

    ExpRecorder() {}
};

}

#endif
