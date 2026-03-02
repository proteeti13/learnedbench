#ifndef NONLEAFNODE3D_H
#define NONLEAFNODE3D_H
#include <vector>
#include "Node.h"

namespace rsmi3dent {

class NonLeafNode : public nodespace3d::Node
{
public:
    int level;
    vector<nodespace3d::Node*> *children;
    NonLeafNode *parent;
    NonLeafNode();
    NonLeafNode(Mbr mbr);
    void addNode(nodespace3d::Node*);
    void addNodes(vector<nodespace3d::Node*>);
    bool is_full();
};

}

#endif
