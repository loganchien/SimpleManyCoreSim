#include "SimConfig.hpp"

#include "Dimension.hpp"

#include <stdlib.h>

SimConfig GlobalConfig;

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

Dim2 SimConfig::int ComputeInCoreBlockIdx1(const Dim2& inCoreBlockIdx2);
{
	return inCoreBlockIdx2.y * CoreBlockLen + inCoreBlockIdx2.x;
}

int SimConfig::GetTotalL2CacheSize()
{
    return CoreBlockSize().Area() * CacheL2Size;
}
