#ifndef MMU_HPP
#define MMU_HPP

/**
 * This file already exists in ARMulator.
 * We have to add modifications in order to simulate our new memory-access
 * system.
 */

#include "Address.hpp"
#include "Cache.hpp"
#include "Message.hpp"

#include <vector>
#include <stdint.h>

namespace smcsim {

/// We can have a lot of concurrently outstanding requests
#define MAX_OUTSTANDING_REQUESTS 8192

class CacheLine;
class Tile;

class OutstandingRequest
{
public:
    Address addr;
    bool pending;
    int totalDelay;
    Dim2 requesterIdx;
	int origRequestId; //TODO: fill in when OutstandingReq is created
};

/// A per-tile MMU
class MMU
{
public:
    /// The tile to which this local MMU belongs
    Tile* tile;

    /// L1 cache and one chunk of a distributed and shared L2 cache
    Cache l1, l2;

    /// The index of this tile's L2 chunk
    int l2ChunkIdx;

    /// The last used request buffer entry index
    int lastRequestId;

    /**
     * Request buffer: Currently outstanding requests, going to an off-tile
     * destination.  This includes requests from the on-tile core, as well as
     * requests from off-tile cores (i.e. off-tile L2 cache misses that happen
     * on this tile).
     */
    std::vector<OutstandingRequest> requests;

    uint32_t loadStallAddr;


    // #################### Simulation stuff ##################################

    /// Total simulated time spent on memory requests
    long long simTime;

public:
    MMU(Tile* tile);

    /// New custom function that we call during start-up
    void InitMMU();


    /// Reset MMU to initial state
    void ResetMMU();


    /// Loads a word into the on-tile Core
    void LoadWord(const Address& addr);


    /// Fetch cacheline from the given off-tile L2 cache chunk. Note that this
    /// is only used by the local Core.
    int FetchRemoteL2(const Dim2& holderIdx, int totalDelay,
                      const Address& addr);


    /// Fetch word from on-tile L2
    void FetchLocalL2(const Dim2& requesterIdx, int requestId, int totalDelay,
                      const Address& addr);


    /// Fetch word from memory, when it is missing in this tile's L2
    int FetchFromMemory(const Dim2& requesterIdx, int requestId, const Address& addr,
                        int totalDelay);


    // #################### Handle incoming Messages ##########################

    /// Called by router when a Cacheline has been sent to this tile
    void OnCachelineReceived(int requestId, int totalDelay,
                             const CacheLine& cacheLine);


    /// This function is always called when the MMU retrieved a word
    void CommitLoad(CacheLine* line, int totalDelay, const Address& addr);


    // #################### Handle Request buffer & Send Messages #############

    /// Get a free request buffer entry
    OutstandingRequest& GetFreeRequest(int& requestId);

    /// Creates and sends a new Request Message
    void SendRequest(MessageType type, const Dim2& requesterIdx, int requestId,
                     const Dim2& receiver, Address addr, int totalDelay);


    /// Creates and sends a new Response Message
    void SendResponse(MessageType type, const Dim2& receiver,
                      const int requestId, const CacheLine &cacheLine,
                      int totalDelay);


    // #################### Interfaces for ARMulator ##########################

    /// Load the byte at the address in the memory.  If the byte can't be
    /// retrived without the stall, then throw LoadStall exception.
    uint8_t GetByte(uint32_t addr, bool simulateDelay);

    /// Load the half word at the address in the memory.  If the half word
    /// can't be retrived without the stall, then throw LoadStall exception.
    uint16_t GetHalfWord(uint32_t addr, bool simulateDelay);

    /// Load the word at the address in the memory.  If the word can't be
    /// retrived without the stall, then throw LoadStall exception.
    uint32_t GetWord(uint32_t addr, bool simulateDelay);

    /// Store the byte at the address in the memory.
    void SetByte(uint32_t addr, uint8_t byte, bool simulateDelay);

    /// Store the half word at the address in the memory.
    void SetHalfWord(uint32_t addr, uint16_t halfword, bool simulateDelay);

    /// Store the word at the address in the memory.
    void SetWord(uint32_t addr, uint32_t word, bool simulateDelay);

    int GetEntry();

    int GetStackTop();

    int GetStackSize();

    int GetHeapTop();

    int GetHeapSize();

private:
    void SimulateLoadStall(bool simulateDelay, uint32_t addr, uint32_t delay);
};

} // end namespace smcsim

#endif // MMU_HPP
