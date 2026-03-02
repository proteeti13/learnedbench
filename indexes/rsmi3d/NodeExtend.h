#ifndef NODEEXTEND3D_H
#define NODEEXTEND3D_H
#include <vector>
#include "Node.h"
#include "Point.h"

namespace rsmi3dent {

class NodeExtend {
public:
    nodespace3d::Node *node = nullptr;
    Point point;
    float dist;
    NodeExtend();
    NodeExtend(nodespace3d::Node*, float);
    NodeExtend(Point, float);
    bool is_leafnode();
};

}

#endif
