#include "Node.h"
#include <math.h>
using namespace std;

namespace rsmi3dent {
namespace nodespace3d {

Node::Node() {}

float Node::cal_dist(Point p)
{
    return mbr.cal_dist(p);
}

}}
