#ifndef MMU_H
#define MMU_H

/**
 * This file already exists in ARMulator. 
 * We have to add modifications in order to simulate our new memory-access system.
 */

/// We can have a lot of concurrently outstanding requests
#define MAX_OUTSTANDING_REQUESTS 8192

#include "Cache.hpp"
#include "VisualCXXCompatibility.hpp"
#include "Message.hpp"
#include "simutil.hpp"

#include <vector>

struct Tile;
struct Processor;

struct OutstandingRequest
{
    Address addr;
    bool pending;
    int totalDelay;
    int2 requesterIdx;
};

/// A per-tile MMU
struct MMU
{
    /// The tile to which this local MMU belongs
    Tile* tile;
    
    /// L1 cache and one chunk of a distributed and shared L2 cache
    Cache l1, l2;

    /// The index of this tile's L2 chunk
    int l2ChunkIdx;

    /// The last used request buffer entry index
    int lastRequestId;
    
    /**
     * Request buffer: Currently outstanding requests, going to an off-tile destination. 
     * This includes requests from the on-tile core, as well as requests from off-tile cores (i.e. off-tile L2 cache misses that happen on this tile).
     */
    std::vector<OutstandingRequest> requests;


    
    // ######### Simulation stuff #########

    /// Total simulated time spent on memory requests
    long long simTime;

    // ...

    /// New custom function that we call during start-up
    void InitMMU(Tile* tile);

    /// Reset MMU to initial state
    void ResetMMU();

    /// Loads a word into the on-tile Core
    void LoadWord(const Address& addr);

    /// Fetch cacheline from the given off-tile L2 cache chunk. Note that this is only used by the local Core.
    int FetchRemoteL2(int2 holderIdx, int totalDelay, const Address& addr);

    /// Fetch word from on-tile L2
    void FetchLocalL2(int2 requesterIdx, int totalDelay, const Address& addr);

    
    /// Fetch word from memory, when it is missing in this tile's L2
    int FetchFromMemory(int2 requesterIdx, const Address& addr, int totalDelay);


    /// Called by router when a Cacheline has been sent to this tile
    void OnCachelineReceived(int requestId, int totalDelay, WORD* words);

    /// This function is always called when the MMU retrieved a word
    void CommitLoad(CacheLine* line, int totalDelay, const Address& addr);


    // ############################################# Handle Request buffer & Send Messages #############################################

    /// Get a free request buffer entry
    OutstandingRequest& GetFreeRequest(int& requestId);

    /// Creates and sends a new Request Message
    void SendRequest(MessageType type, int2 requesterIdx, int2 receiver, Address addr, int totalDelay);

    /// Creates and sends a new Response Message
    void SendResponse(MessageType type, int2 receiver, WORD* words, int totalDelay);
};

#endif // MMU_H
