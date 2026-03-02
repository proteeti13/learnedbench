#ifndef LEAFNODE3D_H
#define LEAFNODE3D_H
#include <vector>
#include "Node.h"
#include "Point.h"
#include "Mbr.h"
#include "NonLeafNode.h"
using namespace std;

namespace rsmi3dent {

class LeafNode : public nodespace3d::Node
{
public:
    int level;
    vector<Point> *children;
    NonLeafNode *parent;

    LeafNode();
    LeafNode(Mbr mbr);

    void add_point(Point p);
    void add_points(vector<Point> pts);
    bool delete_point(Point p);
    bool is_full();
    LeafNode split1();
};

}

#endif
