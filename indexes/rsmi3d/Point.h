#ifndef POINT3D_H
#define POINT3D_H
#include <vector>
#include <string>
using namespace std;

namespace rsmi3dent {

class Point
{
public:
    float index;
    float x;          // SourceID (normalized)
    float y;          // Hop1_ID  (normalized)
    float z;          // Hop2_ID  (normalized)
    long long x_i;    // rank in x-sorted order
    long long y_i;    // rank in y-sorted order
    long long z_i;    // rank in z-sorted order
    long long curve_val;
    float normalized_curve_val;
    float temp_dist = 0.0;

    Point(float x, float y, float z);
    Point();
    bool operator==(const Point& p);
    float cal_dist(Point p);
    void print();
    static vector<Point> get_points(vector<Point> dataset, int num);
    string get_self();
};

}

#endif
