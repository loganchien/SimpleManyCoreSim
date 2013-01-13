#ifndef CACHE_HPP
#define CACHE_HPP

#include <vector>

#include <stddef.h>
#include <stdint.h>

namespace smcsim {

class Cache;

/// A traditional cache line
class CacheLine
{
public:
    /// The Cache this CacheLine belongs to.
    const Cache* owner;

    /// Whether the content in this CacheLine is valid or not.
    bool valid;

    /// The tag to check the mapping between the CacheLine and the Memory.
    uint32_t tag;

    /// The content of the CacheLine.
    std::vector<uint8_t> content;

public:
    CacheLine();

    void InitCacheLine(Cache* owner, size_t cacheLineSize);

    /// Get the word at the given address (given the address maps to this line)
    uint32_t GetWord(uint32_t addr) const;

    /// Set the word at the given address
    void SetWord(uint32_t addr, uint32_t word);

    /// Get the raw contents.
    const uint8_t* GetContent() const
    {
        return &*content.begin();
    }

    /// Set the content of the CacheLine
    void SetLine(uint32_t addr, const uint8_t* newContent);
};


/// A traditional cache or cache chunk
class Cache
{
public:
    /// Cache Lines
    std::vector<CacheLine> lines;

    /// The capacity of the cache
    uint32_t capacity;

    /// The cache line size in bytes
    uint32_t cacheLineSize;

    /// Address space should be handled by this cache
    uint32_t addrSpaceBegin;
    uint32_t addrSpaceEnd; // (inclusive)
    // Note: addrSpaceEnd should be inclusive because we don't want to promote
    // the type to uint64_t.

    /// Amount of all accesses and of misses in this cache (chunk)
    long long simAccessCount;
    long long simMissCount;

public:
    /// Initialize the Cache.
    void InitCache(uint32_t cache, uint32_t cacheLineSize,
                   uint32_t addrSpaceBegin, uint32_t addrSpaceSize);

    /// Reset the Cache.
    void Reset();

    /// Get the CacheLine of the given addr.  If the CacheLine corresponding
    /// to the addr is invalid or having different tag, then return false.
    /// If the CacheLine is the correct one, then set the line,
    /// and return true.
    bool GetLine(uint32_t addr, CacheLine*& line);

    /// Look for the CacheLine with the same index with addr.
    CacheLine& GetSameIndexLine(uint32_t addr);

    /// Set the content of the CacheLine.
    void SetLine(uint32_t addr, const uint8_t *content);

    /// Get the tag of the addr.
    uint32_t GetAddrTag(uint32_t addr) const;

    /// Get the direct mapped index of the addr.
    uint32_t GetAddrIndex(uint32_t addr) const;

    /// Get the offset in the cache line of the addr.
    uint32_t GetAddrOffset(uint32_t addr) const;
};

} // end namespace smcsim

#endif // CACHE_HPP
