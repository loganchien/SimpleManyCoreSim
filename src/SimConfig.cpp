#include "SimConfig.hpp"

#include "Dimension.hpp"

#include <stdlib.h>

Dim2 SimConfig::CoreGridSize()
{
    return Dim2(CoreGridLen, CoreGridLen);
}

Dim2 SimConfig::CoreBlockSize()
{
    return Dim2(CoreBlockLen, CoreBlockLen);
}

Dim2 SimConfig::ComputeInCoreBlockIdx2(int inCoreBlockIdx1)
{
    div_t d = div(inCoreBlockIdx1, CoreBlockLen);
    return Dim2(d.quot, d.rem);
}

int SimConfig::GetTotalL2CacheSize()
{
    return CoreBlockSize().Area() * CacheL2Size;
}
