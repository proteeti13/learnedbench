#include "Point.h"
#include <iostream>
#include <vector>
#include <string>
#include <math.h>
using namespace std;

namespace rsmi3dent {

Point::Point() {}

Point::Point(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

bool Point::operator==(const Point &p)
{
    if (this == &p) return true;
    return (this->x == p.x && this->y == p.y && this->z == p.z);
}

float Point::cal_dist(Point p)
{
    if (temp_dist == 0)
        temp_dist = sqrt(pow(p.x - x, 2) + pow(p.y - y, 2) + pow(p.z - z, 2));
    return temp_dist;
}

void Point::print()
{
    cout << "(x=" << x << ",y=" << y << ",z=" << z << ")"
         << " index=" << index << " curve_val=" << curve_val << endl;
}

vector<Point> Point::get_points(vector<Point> dataset, int num)
{
    srand(time(0));
    int length = dataset.size();
    vector<Point> points;
    for (int i = 0; i < num; i++)
        points.push_back(dataset[rand() % length]);
    return points;
}

string Point::get_self()
{
    return to_string(x) + "," + to_string(y) + "," + to_string(z) + "\n";
}

}
