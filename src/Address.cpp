#include "Address.hpp"

#include "SimConfig.hpp"

uint32_t Address::GetL1Index() const
{
    return 0; // TODO: Not implemented
}

uint32_t Address::GetL1Tag() const
{
    return 0; // TODO: Not implemented
}

uint32_t Address::GetL2Index() const
{
    return 0; // TODO: Not implemented
}

uint32_t Address::GetL2Tag() const
{
    return 0; // TODO: Not implemented
}

/// The offset of the word that this address is referring to
uint32_t Address::GetWordOffset() const
{
    return raw & GlobalConfig.CacheLineBits;
}

/// The block-local L2 chunk index, to which this address maps
int Address::GetL2ChunkIdx1() const
{
    return (GetL2Index() / GlobalConfig.CacheL2Size) % GlobalConfig.CoreBlockSize();
}
