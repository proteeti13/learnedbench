#include "NodeExtend.h"

namespace rsmi3dent {

NodeExtend::NodeExtend() {}
NodeExtend::NodeExtend(nodespace3d::Node* n, float d) { node = n; dist = d; }
NodeExtend::NodeExtend(Point p, float d)               { point = p; dist = d; }
bool NodeExtend::is_leafnode()                          { return node == nullptr; }

}
