#ifndef NODE3D_H
#define NODE3D_H
#include "Mbr.h"

namespace rsmi3dent {

namespace nodespace3d {
class Node
{
public:
    Mbr mbr;
    int order_in_level;
    Node();
    virtual ~Node() {}
    float cal_dist(Point p);
};
};

}

#endif
