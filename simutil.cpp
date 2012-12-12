#include "simutil.hpp"

#include "SimConfig.hpp"

int Address::GetL2ChunkIndex() const
{
    return (L2Index / GlobalConfig.CacheL2Size) % GlobalConfig.CoreBlockSize();
}
