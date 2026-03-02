#include "NonLeafNode.h"
#include "Constants.h"
using namespace rsmi3dutil;

namespace rsmi3dent {

NonLeafNode::NonLeafNode()
{
    children = new vector<nodespace3d::Node*>();
}

NonLeafNode::NonLeafNode(Mbr mbr)
{
    this->mbr = mbr;
    children = new vector<nodespace3d::Node*>();
}

void NonLeafNode::addNode(nodespace3d::Node* node)
{
    children->push_back(node);
    mbr.update(node->mbr);
}

void NonLeafNode::addNodes(vector<nodespace3d::Node*> nodes)
{
    for (auto n : nodes) addNode(n);
}

bool NonLeafNode::is_full()
{
    return (int)children->size() >= Constants::PAGESIZE;
}

}
