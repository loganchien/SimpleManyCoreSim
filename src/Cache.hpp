#ifndef CACHE_HPP
#define CACHE_HPP

#include <vector>
#include <stdint.h>

class Address;

/// A traditional cache line
class CacheLine
{
public:
    bool valid;
    uint32_t tag;
    std::vector<uint8_t> bytes;

public:
    CacheLine();

    /// Get the word at the given address (given the address maps to this line)
    uint32_t GetWord(const Address& addr) const;

    /// Set the word to the given address (given the address maps to this line)
    void SetWord(const Address& addr, uint32_t word);
};


/// A traditional cache or cache chunk
class Cache
{
public:
    // TODO: Add associativity (?)

    /// All lines
    std::vector<CacheLine> lines;

    /// Amount of lines
    int size;

    /// Address offset of this cache chunk. Must be subtracted from actual entry index. Default = 0
    int offset;


    /// Amount of all accesses and of misses in this cache (chunk)
    long long simAccessCount, simMissCount;



    void InitCache(int size, int offset = 0);


    /// The given CacheLine is updated
    CacheLine& UpdateLine(const Address& addr, const CacheLine &cachdLine);

    CacheLine* GetLine(const Address& addr);


    /// Usually the cache is only reset when the processor is reset (i.e. when starting a new batch)
    void Reset();


    /// Get the entry of the given address
    bool GetEntry(const Address& addr, CacheLine* line);
};

#endif // CACHE_HPP
