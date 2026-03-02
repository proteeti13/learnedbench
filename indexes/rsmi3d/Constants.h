#ifndef CONSTANTS3D_H
#define CONSTANTS3D_H
#include <string>
using namespace std;

namespace rsmi3dutil {
class Constants
{
public:
    static const int DIM = 3;          // 3-dimensional points
    static const int PAGESIZE = 100;
    static const int EACH_DIM_LENGTH = 8;
    static const int INFO_LENGTH = 8;
    static const int MAX_WIDTH = 16;
    static const int EPOCH = 500;
    static const int START_EPOCH = 300;
    static const int EPOCH_ADDED = 100;
    static const int HIDDEN_LAYER_WIDTH = 50;
    static const int THRESHOLD = 10000;

    static const int DEFAULT_SIZE  = 16000000;
    static const int DEFAULT_SKEWNESS  = 4;

    static const double LEARNING_RATE;
    Constants();
};
}

#endif
