

#include <vector>

/// A traditional cache line
struct CacheLine
{
    bool valid;
    int tag;
    WORD words[SimConfig::CacheLineWordSize];
    
    CacheLine()
    {
        valid = false;
    }

    /// Get the word at the given address (given the address maps to this line)
    WORD GetWord(const Address& addr)
    {
        return words[addr.WordOffset];
    }
};


/// A traditional cache or cache chunk
struct Cache
{
    // TODO: Add associativity (?)

    /// All lines
    std::vector<CacheLine> lines;

    /// Amount of lines
    int size;
    
    /// Address offset of this cache chunk. Must be subtracted from actual entry index. Default = 0
    int offset;


    /// Amount of all accesses and of misses in this cache (chunk)
    long long simAccessCount, simMissCount;



    void InitCache(int size, int offset = 0)
    {
        this->size = size;
        this->offset = offset;

        lines.resize(size);
    }


    /// The given CacheLine is updated
    CacheLine& UpdateLine(const Address& addr, WORD* words)
    {
        CacheLine& line = lines[addr.Index - offset];
        
        // TODO: Copy words into cacheline and set valid and tag

        return line;
    }

    
    /// Usually the cache is only reset when the processor is reset (i.e. when starting a new batch)
    void Reset()
    {
        lines.clear();
        lines.resize(size);

        simAccessCount = simMissCount = 0;
    }
    

    /// Get the entry of the given address
    bool GetEntry(const Address& addr, CacheLine* line)
    {   
        ++simAccessCount;

        line = &lines[addr.Index - offset];
        if (line->valid && line->tag == addr.tag)
        {
            // cache hit
            return true;
        }
        
        // cache miss
        ++simMissCount;
        return false;
    }
};