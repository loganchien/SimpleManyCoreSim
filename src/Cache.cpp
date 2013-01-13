#include "Cache.hpp"

#include <vector>
#include "Debug.hpp"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

using namespace smcsim;

CacheLine::CacheLine(): owner(0), valid(false), tag(0)
{
}


/// Initialize the CacheLine
void CacheLine::InitCacheLine(Cache* owner_, size_t cacheLineSize)
{
    owner = owner_;
    valid = false;
    tag = 0;
    content.resize(cacheLineSize);
}


/// Get the word at the given address (given the address maps to this line)
uint32_t CacheLine::GetWord(uint32_t addr) const
{
    return *reinterpret_cast<const uint32_t*>(
        &*content.begin() + owner->GetAddrOffset(addr));
}


/// Set the word at the given address (given the address maps to this line)
void CacheLine::SetWord(uint32_t addr, uint32_t word)
{
    *reinterpret_cast<uint32_t*>(
        &*content.begin() + owner->GetAddrOffset(addr)) = word;
}


/// Set the content of the CacheLine
void CacheLine::SetLine(uint32_t addr, const uint8_t* newContent)
{
    valid = true;
    tag = owner->GetAddrTag(addr);
    memcpy(&*content.begin(), newContent, content.size());
}


void Cache::InitCache(uint32_t capacity_, uint32_t cacheLineSize_,
                      uint32_t addrSpaceBegin_, uint32_t addrSpaceSize)
{
    // Initialize the stats
    simAccessCount = 0;
    simMissCount = 0;

    // Initialize the cache capacity and the cache line size
    capacity = capacity_;
    cacheLineSize = cacheLineSize_;
    assert(capacity != 0 && cacheLineSize != 0);

    // Initialize the address space of this cache
    // Note: addrSpaceSize = 0 stands for addrSpaceSize = 4G.
    addrSpaceBegin = addrSpaceBegin_;
    addrSpaceEnd = addrSpaceBegin_ + addrSpaceSize - 1;
    assert(addrSpaceBegin <= addrSpaceEnd);

    // Initialize CacheLine
    assert(capacity > cacheLineSize && capacity % cacheLineSize == 0);
    lines.resize(capacity / cacheLineSize);

    for (size_t i = 0; i < lines.size(); ++i)
    {
        lines[i].InitCacheLine(this, cacheLineSize);
    }
}


void Cache::Reset()
{
    simAccessCount = 0;
    simMissCount = 0;
    capacity = 0;
    cacheLineSize = 0;
    addrSpaceBegin = 0;
    addrSpaceEnd = 0;
    simAccessCount = 0;
    simMissCount = 0;

    lines.clear();
}


/// Get the entry of the given address
bool Cache::GetLine(uint32_t addr, CacheLine*& line)
{
    // Find the cache line
    CacheLine& foundLine = GetSameIndexLine(addr);
    if (foundLine.valid && foundLine.tag == GetAddrTag(addr))
    {
        // Cache Hit
        line = &foundLine;
        return true;
    }
    return false;
}


/// The given CacheLine is updated
void Cache::SetLine(uint32_t addr, const uint8_t *content)
{
    CacheLine& line = GetSameIndexLine(addr);
    line.SetLine(addr, content);
}


CacheLine& Cache::GetSameIndexLine(uint32_t addr)
{
    return lines[GetAddrIndex(addr)];
}


/// Get the tag of the addr.
uint32_t Cache::GetAddrTag(uint32_t addr) const
{
    return (addr / cacheLineSize);
}


/// Get the direct mapped index of the addr.
uint32_t Cache::GetAddrIndex(uint32_t addr) const
{
    assert(addr >= addrSpaceBegin && addr <= addrSpaceEnd);
    return (((addr - addrSpaceBegin) / cacheLineSize) % lines.size());
}


/// Get the offset in the cache line of the addr.
uint32_t Cache::GetAddrOffset(uint32_t addr) const
{
    return (addr % cacheLineSize);
}
