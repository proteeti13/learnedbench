#ifndef SORTTOOLS3D_H
#define SORTTOOLS3D_H

#include <vector>
#include <algorithm>
#include "./Constants.h"
#include "./Mbr.h"
#include "./Point.h"
#include "./NodeExtend.h"

using namespace std;
using namespace rsmi3dent;

namespace rsmi3dutil {

struct sortPQ
{
    bool operator()(const rsmi3dent::NodeExtend *n1, const rsmi3dent::NodeExtend *n2)
    { return n1->dist > n2->dist; }
};

struct sortForKNN
{
    rsmi3dent::Point queryPoint;
    sortForKNN(rsmi3dent::Point& p) { queryPoint = p; }
    bool operator()(rsmi3dent::Point p1, rsmi3dent::Point p2)
    { return p1.cal_dist(queryPoint) < p2.cal_dist(queryPoint); }
};

struct sortForKNN2
{
    bool operator()(rsmi3dent::Point p1, rsmi3dent::Point p2)
    { return p1.temp_dist > p2.temp_dist; }
};

struct sortX
{
    bool operator()(const rsmi3dent::Point p1, const rsmi3dent::Point p2)
    { return p1.x < p2.x; }
};

struct sortY
{
    bool operator()(const rsmi3dent::Point p1, const rsmi3dent::Point p2)
    { return p1.y < p2.y; }
};

struct sortZ
{
    bool operator()(const rsmi3dent::Point p1, const rsmi3dent::Point p2)
    { return p1.z < p2.z; }
};

struct sort_curve_val
{
    bool operator()(const rsmi3dent::Point p1, const rsmi3dent::Point p2)
    { return p1.curve_val < p2.curve_val; }
};

}

#endif
