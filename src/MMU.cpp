#include "MMU.hpp"

#include "Core.hpp"
#include "CoreBlock.hpp"
#include "Debug.hpp"
#include "Dimension.hpp"
#include "GlobalMemoryController.hpp"
#include "Processor.hpp"
#include "SimConfig.hpp"
#include "Task.hpp"
#include "TaskBlock.hpp"
#include "Thread.hpp"
#include "Tile.hpp"

#include "ArmulatorError.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

using namespace smcsim;

MMU::MMU(Tile* tile_)
    : tile(tile_), lastLoadStallAddr(0), lastLoadStallResult(0)
{ }

/// New custom function that we call during start-up
void MMU::InitMMU()
{
    // Initialize Caches
    l1.InitCache(GlobalConfig.CacheL1Size, GlobalConfig.CacheLineSize, 0, 0);
    l2.InitCache(GlobalConfig.CacheL2Size, GlobalConfig.CacheLineSize, 0, 0);

    //l2ChunkIdx = tile->coreBlock->ComputeL2ChunkID(tile->tileIdx);
    //PrintLine("MMU: ChunkIdx: " << l2ChunkIdx);
    //l2.InitCache(GlobalConfig.CacheL2Size, l2ChunkIdx * GlobalConfig.CacheL2Size);
}


/// Reset MMU to initial state
void MMU::ResetMMU()
{
    // Clear caches & reset simulated time
    l1.Reset();
    l2.Reset();

    simTime = 0;

    // Clear outstanding requests
    lastRequestId = 0;
    requests.clear();
    requests.resize(MAX_OUTSTANDING_REQUESTS);
}


/// Loads a word into the on-tile Core
void MMU::LoadWord(const Address& addr)
{
    assert(0 && "Not implemented.  Currently, please call GetWord instead");
    abort();

#if 0
    // Lookup in caches
    // L1 access time
    int totalDelay = GlobalConfig.CacheL1Delay;

    CacheLine* line = 0;
    if (!l1.GetLine(addr.raw, line))
    {
        // L1 miss
        int addrL2ChunkIdx = tile->coreBlock->ComputeL2ChunkID(addr);
        if (addrL2ChunkIdx != l2ChunkIdx)
        {
            // Get from off-tile L2
            FetchRemoteL2(tile->coreBlock->ComputeTileIndex(addrL2ChunkIdx), totalDelay, addr);
        }
        else
        {
            // Get from local L2
            FetchLocalL2(tile->tileIdx, -1, totalDelay, addr);
        }
    }
    else
    {
        CommitLoad(line, totalDelay, addr);
    }
#endif
}


/// Fetch cacheline from the given off-tile L2 cache chunk. Note that this is only used by the local Core.
int MMU::FetchRemoteL2(const Dim2& holderIdx, int totalDelay,
                       const Address& addr)
{
    SendRequest(MessageTypeRequestL2, tile->tileIdx, -1, holderIdx, addr, totalDelay);
}


/// Fetch word from on-tile L2
void MMU::FetchLocalL2(const Dim2& requesterIdx, int requestId, int totalDelay,
                       const Address& addr)
{
    // Add hit penalty
    totalDelay += GlobalConfig.CacheL2Delay;

    CacheLine* line = NULL;
    if (!l2.GetLine(addr.raw, line))
    {
        // L2 miss
        FetchFromMemory(requesterIdx, requestId, addr, totalDelay);
    }
    else
    {
        if (requesterIdx == tile->tileIdx)
        {
            // Local request -> Can be committed immediately
            CommitLoad(line, totalDelay, addr);
        }
        else
        {
            // Send value back to requester
            SendResponse(MessageTypeResponseCacheline, requesterIdx, requestId, *line, totalDelay);
        }
    }
}


/// Fetch word from memory, when it is missing in this tile's L2
int MMU::FetchFromMemory(const Dim2& requesterIdx, int requestId,
                         const Address& addr, int totalDelay)
{
    SendRequest(MessageTypeRequestMem, requesterIdx, requestId, Dim2(GlobalMemoryController::GMemIdx, GlobalMemoryController::GMemIdx), addr, totalDelay);
}


// #################### Handle incoming Messages ##############################

/// Called by router when a Cacheline has been sent to this tile
void MMU::OnCachelineReceived(int requestId, int totalDelay,
                              const CacheLine &cacheLine)
{
    OutstandingRequest& request = requests[requestId];
    assert(request.pending);

    int addrChunkIdx = request.addr.GetL2Index() / GlobalConfig.CacheL2Size;
    if (addrChunkIdx == l2ChunkIdx)
    {
        // Address maps to this tile's L2 chunk, so have to update it
        l2.SetLine(request.addr.raw, cacheLine.GetContent());
    }

    // AdvTODO: Handle coalescing (i.e. multiple requesters request the same
    // line in a single packets)

    if (request.requesterIdx == tile->tileIdx)
    {
        // Cacheline was requested by this guy

        // -> Also goes into L1
        l1.SetLine(request.addr.raw, cacheLine.GetContent());

        // And we can finally restart the CPU
        CommitLoad(&l1.GetSameIndexLine(request.addr.raw),
                   totalDelay, request.addr);
    }
    else
    {
        // CacheLine is a response to an off-tile request -> Send it to
        // requester
        SendResponse(MessageTypeResponseCacheline, request.requesterIdx,
                     request.origRequestId, cacheLine, totalDelay);
    }

    // Request buffer entry not in use anymore
    request.pending = false;
}


/// This function is always called when the MMU retrieved a word
void MMU::CommitLoad(CacheLine* line, int totalDelay, const Address& addr)
{
    // Add time spent on this request to total sim time
    simTime += totalDelay;
    tile->core.CommitLoad(line->GetWord(addr.raw));
}



// #################### Handle Request buffer & Send Messages #################

/// Get a free request buffer entry
OutstandingRequest& MMU::GetFreeRequest(int& requestId)
{
    for (int i = lastRequestId, count = 0; count < MAX_OUTSTANDING_REQUESTS;
         i = (i+1)%MAX_OUTSTANDING_REQUESTS)
    {
        if (!requests[i].pending)
        {
            requestId = i;
            return requests[i];
        }
    }

    PrintLine("More than MAX_OUTSTANDING_REQUESTS in request queue");
    assert(false);
}

/// Creates and sends a new Request Message
void MMU::SendRequest(MessageType type, const Dim2& requesterIdx, int origReqId,
                      const Dim2& receiver, Address addr, int totalDelay)
{
    // Add request to request buffer
    int requestId;
    OutstandingRequest& request = GetFreeRequest(requestId);
    request.pending = true;
    request.requesterIdx = requesterIdx;
    request.addr = addr;
    request.totalDelay = totalDelay;

    // Create Message object
    Message msg;
    msg.type = type;
    msg.sender = tile->tileIdx;
    msg.receiver = receiver;
    msg.requestId = requestId;
    msg.totalDelay = totalDelay;
    msg.addr = addr;

    // Send the message
    tile->router.EnqueueMessage(msg);
}


/// Creates and sends a new Response Message
void MMU::SendResponse(MessageType type, const Dim2& receiver, const int reqId,
                       const CacheLine& cacheLine, int totalDelay)
{
    // Create Message object
    Message msg;
    msg.type = type;
    msg.sender = tile->tileIdx;
    msg.receiver = receiver;
    msg.requestId = reqId; // FIXME: TODO: requestId
    msg.totalDelay = totalDelay;
    msg.addr = Address();
    msg.cacheLine = cacheLine;

    // Send the message
    tile->router.EnqueueMessage(msg);
}


/// Load the byte at the address in the memory.  If the byte can't be
/// retrived without the stall, then throw LoadStall exception.
uint8_t MMU::GetByte(uint32_t addr, bool simulateDelay)
{
    uint32_t align = addr & ~0x3u;
    uint32_t shift = addr & 0x3u;
    uint32_t word = GetWord(align, simulateDelay);
    return (word >> (shift * 8));
}


/// Load the half word at the address in the memory.  If the half word
/// can't be retrived without the stall, then throw LoadStall exception.
uint16_t MMU::GetHalfWord(uint32_t addr, bool simulateDelay)
{
    assert((addr & 0x1u) == 0);
    uint32_t align = addr & ~0x3u;
    uint32_t shift = addr & 0x3u;
    uint32_t word = GetWord(align, simulateDelay);
    return (word >> (shift * 8));
}


/// Load the word at the address in the memory.  If the word can't be
/// retrived without the stall, then throw LoadStall exception.
uint32_t MMU::GetWord(uint32_t addr, bool simulateDelay)
{
    // Inject the special variables
    Thread* thread = tile->core.currentThread;
    TaskBlock* taskBlock = thread->taskBlock;
    Task* task = taskBlock->task;

#define VAR_VALUE_MAP(ADDR, VAR) \
    do { if (addr == (ADDR)) { return (VAR); } } while (0)

    VAR_VALUE_MAP(task->threadIdxAddr + 0, thread->threadIdx.y);
    VAR_VALUE_MAP(task->threadIdxAddr + 4, thread->threadIdx.x);
    VAR_VALUE_MAP(task->threadDimAddr + 0, task->threadDim.y);
    VAR_VALUE_MAP(task->threadDimAddr + 4, task->threadDim.x);
    VAR_VALUE_MAP(task->blockIdxAddr + 0, taskBlock->taskBlockIdx.y);
    VAR_VALUE_MAP(task->blockIdxAddr + 4, taskBlock->taskBlockIdx.x);
    VAR_VALUE_MAP(task->blockDimAddr + 0, task->blockDim.y);
    VAR_VALUE_MAP(task->blockDimAddr + 4, task->blockDim.x);

#undef VAR_VALUE_MAP

    // Bypass the load stall simulation if simulateDelay=false.
    GlobalMemoryController& gmc =
        tile->coreBlock->processor->gMemController;

    if (!simulateDelay)
    {
        return gmc.GetWord(addr, tile);
    }

    // Return the previously stall word
    if (addr == lastLoadStallAddr)
    {
        uint32_t word = lastLoadStallResult;
        lastLoadStallAddr = 0;
        lastLoadStallResult = 0;
        return word;
    }

    // Lookup in caches
    CacheLine* line = NULL;

    // Increase the access latency
    task->Stats.TotalSimulationTime.TotalCount += GlobalConfig.CacheL1Delay;

    // Update cache access counter
    ++l1.simAccessCount;

    if (l1.GetLine(addr, line))
    {
        uint32_t word = line->GetWord(addr);
        SimulateLoadStall(addr, word, simulateDelay,
                          GlobalConfig.CacheL1Delay +
                          GlobalConfig.MemDelay);
        return word;
    }
    else
    {
        // Increase the access latency
        task->Stats.TotalSimulationTime.TotalCount += GlobalConfig.CacheL2Delay;

        // Update cache access counter
        ++l1.simMissCount;
        ++l2.simAccessCount;

        if (l2.GetLine(addr, line))
        {
            CacheLine& L1Line = l1.GetSameIndexLine(addr);
            L1Line.SetLine(addr, line->GetContent());
            uint32_t word = L1Line.GetWord(addr);
            SimulateLoadStall(addr, word, simulateDelay,
                              GlobalConfig.CacheL1Delay +
                              GlobalConfig.MemDelay);
            return word;
        }
        else
        {
            // Update cache access counter
            ++l2.simMissCount;

            // Increase the access latency
            task->Stats.TotalSimulationTime.TotalCount +=
                GlobalConfig.MemDelay;

            CacheLine& L1Line = l1.GetSameIndexLine(addr);
            CacheLine& L2Line = l2.GetSameIndexLine(addr);

            // Read the memory and copy the data to L2 cache line
            uint32_t alignedAddr = addr / l2.cacheLineSize * l2.cacheLineSize;

            gmc.FillCacheLine(alignedAddr, L2Line, tile);
            L2Line.tag = l2.GetAddrTag(addr);
            L2Line.valid = true;

            L1Line.SetLine(addr, L2Line.GetContent());
            uint32_t word = L1Line.GetWord(addr);
            SimulateLoadStall(addr, word, simulateDelay,
                              GlobalConfig.CacheL1Delay +
                              GlobalConfig.CacheL2Delay +
                              GlobalConfig.MemDelay);
            return word;
        }
    }
}

/// Store the byte at the address in the memory.
void MMU::SetByte(uint32_t addr, uint8_t byte, bool simulateDelay)
{
    uint32_t alignAddr = addr & ~0x3u;
    uint32_t word = GetWord(alignAddr, false);

    switch (addr & 0x3u)
    {
    case 0x0u:
        SetWord(alignAddr, (word & 0xffffff00u) | byte, false);
        break;

    case 0x1u:
        SetWord(alignAddr, (word & 0xffff00ffu) | (byte << 8), false);
        break;

    case 0x2u:
        SetWord(alignAddr, (word & 0xff00ffffu) | (byte << 16), false);
        break;

    case 0x3u:
        SetWord(alignAddr, (word & 0x00ffffffu) | (byte << 24), false);
        break;

    default:
        abort();
    }
}

/// Store the half word at the address in the memory.
void MMU::SetHalfWord(uint32_t addr, uint16_t halfword, bool simulateDelay)
{
    uint32_t alignAddr = addr & ~0x3u;
    uint32_t word = GetWord(alignAddr, false);

    switch (addr & 0x3u)
    {
    case 0x0u:
        SetWord(alignAddr, (word & 0xffff0000u) | halfword, false);
        break;

    case 0x2u:
        SetWord(alignAddr, (word & 0x0000ffffu) | (halfword << 16), false);
        break;

    default:
        assert(0 && "Unaligned half-word access");
        abort();
    }
}

/// Store the word at the address in the memory.
void MMU::SetWord(uint32_t addr, uint32_t word, bool simulateDelay)
{
    assert(!simulateDelay && "Simulate delay for store is not implemented");

    // Cache coherence
    CacheLine *line = NULL;
    if (l1.GetLine(addr, line))
    {
        line->SetWord(addr, word);
    }
    if (l2.GetLine(addr, line))
    {
        line->SetWord(addr, word);
    }

    // Memory write back
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    gmc.SetWord(addr, word, tile);
}

void MMU::SimulateLoadStall(uint32_t addr, uint32_t word,
                            bool simulateDelay, uint32_t delay)
{
    if (simulateDelay && delay > 1)
    {
        // The CPU will execute the load instruction again after (MEM_DELAY
        // - 2) clock so that it can load the data.
        lastLoadStallAddr = addr;
        lastLoadStallResult = word;
        throw LoadStall(delay - 2);
    }
}

int MMU::GetEntry()
{
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    return static_cast<int>(gmc.entryPointVMA);
}

int MMU::GetStackTop()
{
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    return gmc.stackBeginVMA + gmc.stackBeginSize;
}

int MMU::GetStackSize()
{
    return GlobalConfig.StackSize;
}

int MMU::GetHeapTop()
{
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    return static_cast<int>(gmc.heapVMA);
}

int MMU::GetHeapSize()
{
    return GlobalConfig.HeapSize;
}
