#ifndef MBR3D_H
#define MBR3D_H
#include <limits>
#include <vector>
#include "Point.h"
using namespace std;

namespace rsmi3dent {

class Mbr
{
public:
    float x1 = numeric_limits<float>::max();
    float x2 = numeric_limits<float>::lowest();
    float y1 = numeric_limits<float>::max();
    float y2 = numeric_limits<float>::lowest();
    float z1 = numeric_limits<float>::max();
    float z2 = numeric_limits<float>::lowest();

    Mbr();
    Mbr(float x1, float y1, float z1, float x2, float y2, float z2);

    void update(float x, float y, float z);
    void update(Point p);
    void update(Mbr mbr);

    bool contains(Point p);
    bool strict_contains(Point p);
    bool interact(Mbr mbr);

    float cal_dist(Point p);
    void print();

    vector<Point> get_corner_points();          // 8 corners (2^3)
    static Mbr get_mbr(Point p, float side);    // axis-aligned cube around p

    void clean();
    string get_self();
};

}

#endif
