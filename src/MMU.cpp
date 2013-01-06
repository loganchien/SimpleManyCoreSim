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

#include <assert.h>
#include <stdint.h>

using namespace smcsim;

MMU::MMU(Tile* tile_): tile(tile_)
{ }

/// New custom function that we call during start-up
void MMU::InitMMU()
{
    // Initialize Caches
    //l2ChunkIdx = tile->coreBlock->ComputeL2ChunkID(tile->tileIdx);

    //l1.InitCache(GlobalConfig.CacheL1Size);
    //l2.InitCache(GlobalConfig.CacheL2Size, l2ChunkIdx * GlobalConfig.CacheL2Size);

    //ResetMMU();
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
    // Lookup in caches

    // L1 access time
    int totalDelay = GlobalConfig.CacheL1Delay;

    CacheLine* line = l1.GetLine(addr);
    if (!line)
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

    CacheLine* line = l2.GetLine(addr);
    if (!line)
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


// ############################################# Handle incoming Messages #############################################

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
        l2.UpdateLine(request.addr, cacheLine);
    }


    // AdvTODO: Handle coalescing (i.e. multiple requesters request the same line in a single packets)

    if (request.requesterIdx == tile->tileIdx)
    {
        // Cacheline was requested by this guy

        // -> Also goes into L1
        CacheLine& line = l1.UpdateLine(request.addr, cacheLine);

        // And we can finally restart the CPU
        CommitLoad(&line, totalDelay, request.addr);
    }
    else
    {
        // CacheLine is a response to an off-tile request -> Send it to requester
        SendResponse(MessageTypeResponseCacheline, request.requesterIdx, request.origRequestId, cacheLine, totalDelay);
    }

    // Request buffer entry not in use anymore
    request.pending = false;
}


/// This function is always called when the MMU retrieved a word
void MMU::CommitLoad(CacheLine* line, int totalDelay, const Address& addr)
{
    // Add time spent on this request to total sim time
    simTime += totalDelay;
    tile->core.CommitLoad(line->GetWord(addr));
}



// ############################################# Handle Request buffer & Send Messages #############################################

/// Get a free request buffer entry
OutstandingRequest& MMU::GetFreeRequest(int& requestId)
{
    for (int i = lastRequestId, count = 0; count < MAX_OUTSTANDING_REQUESTS; i = (i+1)%MAX_OUTSTANDING_REQUESTS)
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


/// Load the byte at the address in the memory without stall.  If the byte
/// can't be retrived without the stall, then return false.
bool MMU::LoadReadyByte(uint32_t addr, uint8_t& byte)
{
    bool ready = true;
    // TODO: Should look for cache line instead
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    byte = gmc.LoadByte(addr);
    return ready;
}


/// Load the half word at the address in the memory without stall.  If the
/// half word can't be retrived without the stall, then return false.
bool MMU::LoadReadyHalfWord(uint32_t addr, uint16_t& halfword)
{
    bool ready = true;
    // TODO: Should look for cache line instead
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    halfword = gmc.LoadHalfWord(addr);
    return ready;
}


/// Load the word at the address in the memory without stall.  If the word
/// can't be retrived without the stall, then return false.
bool MMU::LoadReadyWord(uint32_t addr, uint32_t& word)
{
    // If the addr is mapped to special range, then return the special value,
    // such as threadIdx, threadDim, blockIdx, blockDim.
    Thread* thread = tile->core.currentThread;
    TaskBlock* taskBlock = thread->taskBlock;
    Task* task = taskBlock->task;

#define VAR_VALUE_MAP(ADDR, VAR, SIZE)                                        \
    if (addr >= (ADDR) && addr < ((ADDR) + (SIZE)))                           \
    {                                                                         \
        uint32_t* var = reinterpret_cast<uint32_t*>(&(VAR));                  \
        word = var[(addr - (ADDR)) / sizeof(uint32_t)];                       \
        return true;                                                          \
    }

    VAR_VALUE_MAP(task->threadIdxAddr, thread->threadIdx, sizeof(Dim2));
    VAR_VALUE_MAP(task->threadDimAddr, task->threadDim, sizeof(Dim2));
    VAR_VALUE_MAP(task->blockIdxAddr, taskBlock->taskBlockIdx, sizeof(Dim2));
    VAR_VALUE_MAP(task->blockDimAddr, task->blockDim, sizeof(Dim2));

#undef VAR_VALUE_MAP

    // Load the real values
    bool ready = true;
    // TODO: Should look for cache line instead
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    word = gmc.LoadWord(addr);
    return ready;
}

/// Store the byte at the address in the memory.
void MMU::StoreByte(uint32_t addr, uint8_t byte)
{
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    gmc.StoreByte(addr, byte);
}

/// Store the half word at the address in the memory.
void MMU::StoreHalfWord(uint32_t addr, uint16_t halfword)
{
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    gmc.StoreHalfWord(addr, halfword);
}

/// Store the word at the address in the memory.
void MMU::StoreWord(uint32_t addr, uint32_t word)
{
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    gmc.StoreWord(addr, word);
}

int MMU::GetEntry()
{
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    return static_cast<int>(gmc.entryPointVMA);
}

int MMU::GetStackTop()
{
    GlobalMemoryController& gmc = tile->coreBlock->processor->gMemController;
    // Note: Stack top is at the high address side, so we have to "+1".
    return gmc.stackBeginVMA +
           GlobalConfig.StackSize * (tile->GetGlobalLinearIndex() + 1);
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
