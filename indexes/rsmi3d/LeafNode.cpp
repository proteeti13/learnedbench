#include "LeafNode.h"
#include "Constants.h"
#include <algorithm>
using namespace std;
using namespace rsmi3dutil;

namespace rsmi3dent {

LeafNode::LeafNode()
{
    children = new vector<Point>();
}

LeafNode::LeafNode(Mbr mbr)
{
    this->mbr = mbr;
    children = new vector<Point>();
}

void LeafNode::add_point(Point p)
{
    children->push_back(p);
    mbr.update(p.x, p.y, p.z);   // 3-D MBR update
}

void LeafNode::add_points(vector<Point> pts)
{
    for (auto& p : pts) add_point(p);
}

bool LeafNode::is_full()
{
    return (int)children->size() >= Constants::PAGESIZE;
}

LeafNode LeafNode::split1()
{
    LeafNode right;
    right.parent = this->parent;
    int mid = Constants::PAGESIZE / 2;
    vector<Point> vec(children->begin() + mid, children->end());
    right.add_points(vec);
    vector<Point> vec1(children->begin(), children->begin() + mid);
    children->clear();
    add_points(vec1);
    return right;
}

bool LeafNode::delete_point(Point p)
{
    auto it = find(children->begin(), children->end(), p);
    if (it != children->end()) {
        children->erase(it);
        if (!mbr.strict_contains(p)) {
            mbr.clean();
            for (auto& pt : *children) mbr.update(pt.x, pt.y, pt.z);
        }
        return true;
    }
    return false;
}

}
