#include "Cache.hpp"

#include "Address.hpp"
#include "SimConfig.hpp"
#include <math.h>
#include <vector>
#include <stdint.h>

using namespace smcsim;

CacheLine::CacheLine(): valid(false), tag(0), bytes(GlobalConfig.CacheLineSize)
{
}

/// Get the word at the given address (given the address maps to this line)
uint32_t CacheLine::GetWord(const Address& addr) const
{
    return *reinterpret_cast<const uint32_t*>(&*bytes.begin() +
                                              addr.GetWordOffset());
}

/// Set the word to the given address (given the address maps to this line)
void CacheLine::SetWord(const Address& addr, uint32_t word)
{
    *reinterpret_cast<uint32_t*>(&*bytes.begin() +
                                 addr.GetWordOffset()) = word;
}


void Cache::InitCache(int size, int offset)
{
    this->size = size;
    this->offset = offset;

    lines.resize(size);
}


/// The given CacheLine is updated
CacheLine& Cache::UpdateLine(const Address& addr, const CacheLine &cacheLine)
{
    CacheLine& line = lines[addr.GetL1Index() - offset];

    // TODO: Copy words into cacheline and set valid and tag
    line = cacheLine;

    return line;
}


CacheLine* Cache::GetLine(const Address& addr)
{
	return &lines[addr.GetL1Index() - offset];
}


/// Usually the cache is only reset when the processor is reset (i.e. when starting a new batch)
void Cache::Reset()
{
    lines.clear();
    lines.resize(size);

    simAccessCount = simMissCount = 0;
}


/// Get the entry of the given address
bool Cache::GetEntry(const Address& addr, CacheLine* line)
{
    ++simAccessCount;
    line = &lines[addr.GetIndex(GetIndexMask()) - offset];
    if (line->valid && line->tag == addr.GetTag(GetTagShift()))
    {
        // cache hit
        return true;
    }

    // cache miss
    ++simMissCount;
    return false;
}

int Cache::GetIndexMask(){
	return size*GlobalConfig.CacheLineSize-1;
}
int Cache::GetTagShift(){
	return SimConfig::numbCacheLineBits + int(log(size)/log(2));
}
