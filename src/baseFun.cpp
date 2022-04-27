#include "baseFun.h"

float base::round(float src, const int bits)
{
    //std::cout.precision(2);
    std::stringstream ss;
    float f;
    ss << std::fixed << std::setprecision(bits) << src;
    ss >> f;

    return f;

}