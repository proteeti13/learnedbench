#include "Mbr.h"
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <math.h>
using namespace std;

namespace rsmi3dent {

Mbr::Mbr() {}

Mbr::Mbr(float x1, float y1, float z1, float x2, float y2, float z2)
{
    this->x1 = x1; this->y1 = y1; this->z1 = z1;
    this->x2 = x2; this->y2 = y2; this->z2 = z2;
}

void Mbr::update(float x, float y, float z)
{
    if (x < x1) x1 = x;
    if (x > x2) x2 = x;
    if (y < y1) y1 = y;
    if (y > y2) y2 = y;
    if (z < z1) z1 = z;
    if (z > z2) z2 = z;
}

void Mbr::update(Point p)
{
    update(p.x, p.y, p.z);
}

void Mbr::update(Mbr mbr)
{
    if (mbr.x1 < x1) x1 = mbr.x1;
    if (mbr.x2 > x2) x2 = mbr.x2;
    if (mbr.y1 < y1) y1 = mbr.y1;
    if (mbr.y2 > y2) y2 = mbr.y2;
    if (mbr.z1 < z1) z1 = mbr.z1;
    if (mbr.z2 > z2) z2 = mbr.z2;
}

bool Mbr::contains(Point p)
{
    return (x1 <= p.x && p.x <= x2 &&
            y1 <= p.y && p.y <= y2 &&
            z1 <= p.z && p.z <= z2);
}

bool Mbr::strict_contains(Point p)
{
    return (x1 < p.x && p.x < x2 &&
            y1 < p.y && p.y < y2 &&
            z1 < p.z && p.z < z2);
}

// Two 3D AABBs intersect iff they overlap in all three dimensions.
bool Mbr::interact(Mbr mbr)
{
    if (x2 < mbr.x1 || mbr.x2 < x1) return false;
    if (y2 < mbr.y1 || mbr.y2 < y1) return false;
    if (z2 < mbr.z1 || mbr.z2 < z1) return false;
    return true;
}

// Minimum distance from point to 3D AABB.
float Mbr::cal_dist(Point p)
{
    if (contains(p)) return 0.0f;
    float dx = (p.x < x1) ? (x1 - p.x) : (p.x > x2) ? (p.x - x2) : 0.0f;
    float dy = (p.y < y1) ? (y1 - p.y) : (p.y > y2) ? (p.y - y2) : 0.0f;
    float dz = (p.z < z1) ? (z1 - p.z) : (p.z > z2) ? (p.z - z2) : 0.0f;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

void Mbr::print()
{
    cout << "(x=[" << x1 << "," << x2 << "]"
         << " y=[" << y1 << "," << y2 << "]"
         << " z=[" << z1 << "," << z2 << "])" << endl;
}

// Returns all 8 corners of the 3D bounding box.
vector<Point> Mbr::get_corner_points()
{
    return {
        Point(x1,y1,z1), Point(x2,y1,z1), Point(x1,y2,z1), Point(x2,y2,z1),
        Point(x1,y1,z2), Point(x2,y1,z2), Point(x1,y2,z2), Point(x2,y2,z2)
    };
}

// Returns a cube of half-side `side` centred on point p, clamped to [0,1].
Mbr Mbr::get_mbr(Point p, float side)
{
    float lx1 = p.x - side, lx2 = p.x + side;
    float ly1 = p.y - side, ly2 = p.y + side;
    float lz1 = p.z - side, lz2 = p.z + side;
    if (lx1 < 0) lx1 = 0; if (lx2 > 1) lx2 = 1;
    if (ly1 < 0) ly1 = 0; if (ly2 > 1) ly2 = 1;
    if (lz1 < 0) lz1 = 0; if (lz2 > 1) lz2 = 1;
    return Mbr(lx1, ly1, lz1, lx2, ly2, lz2);
}

void Mbr::clean()
{
    x1 = x2 = y1 = y2 = z1 = z2 = 0;
}

string Mbr::get_self()
{
    return to_string(x1)+","+to_string(y1)+","+to_string(z1)+","
          +to_string(x2)+","+to_string(y2)+","+to_string(z2)+"\n";
}

}
