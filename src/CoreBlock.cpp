#include "CoreBlock.hpp"

#include "Address.hpp"
#include "Core.hpp"
#include "Debug.hpp"
#include "Dimension.hpp"
#include "Processor.hpp"
#include "SimConfig.hpp"
#include "TaskBlock.hpp"
#include "Thread.hpp"
#include "Tile.hpp"

#include <assert.h>
#include <stddef.h>

using namespace smcsim;

CoreBlock::CoreBlock()
    : coreBlockSize(GlobalConfig.CoreBlockSize()), runningTaskBlock(NULL)
{
    tiles = new Tile[coreBlockSize.Area()];
}

CoreBlock::~CoreBlock()
{
    delete [] tiles;
}


/// Initializes this CoreBlock
void CoreBlock::InitCoreBlock(Processor* processor_, const Dim2& blockIdx_)
{
    blockIdx = blockIdx_;
    processor = processor_;
    runningTaskBlock = NULL;

    for (int i = 0; i < coreBlockSize.Area(); ++i)
    {
        Dim2 localTileIdx(Dim2::FromLinear(coreBlockSize, i));
        Dim2 globalTileIdx(blockIdx.y * coreBlockSize.y + localTileIdx.y,
                           blockIdx.x * coreBlockSize.x + localTileIdx.x);
        tiles[i].InitTile(this, globalTileIdx);
    }
}


/// The index of the first tile within this core block
Dim2 CoreBlock::ComputeCoreBlockOrigin() const
{
    return blockIdx * processor->coreBlockSize;
}


/// Computes the global 2D index of the given block-local 1D index
Dim2 CoreBlock::ComputeTileIndex(int inCoreID) const
{
    return ComputeCoreBlockOrigin() + GlobalConfig.ComputeInCoreBlockIdx2(inCoreID);
}

/// Computes the L2 chunk index of the given memory address
int CoreBlock::ComputeL2ChunkID(const Address& addr) const
{
    return ComputeL2ChunkID(GlobalConfig.ComputeInCoreBlockIdx2(addr.GetL2ChunkIdx1()));
}


/// Computes the L2 chunk index of the given global tile index
int CoreBlock::ComputeL2TileChunkID(const Dim2& globalIdx) const
{
    Dim2 blockOrigin = ComputeCoreBlockOrigin();
    return ComputeL2ChunkID(globalIdx - blockOrigin);
}


/// Computes the L2 chunk ID of the given block-local 2D index
int CoreBlock::ComputeL2ChunkID(const Dim2& inCoreBlockIdx2) const
{
    // Converts to 1D index, using some order
    return GlobalConfig.ComputeInCoreBlockIdx1(inCoreBlockIdx2);
}


/// Get the Tile by tileIdx
Tile& CoreBlock::GetLocalTile(const Dim2& localTileIdx)
{
    return tiles[Dim2::ToLinear(coreBlockSize, localTileIdx)];
}


/// Get the Tile by tileIdx
const Tile& CoreBlock::GetLocalTile(const Dim2& localTileIdx) const
{
    return tiles[Dim2::ToLinear(coreBlockSize, localTileIdx)];
}


/// Put the next thread of the given TaskBlock on the given tile
void CoreBlock::ScheduleThread(TaskBlock& taskBlock, Tile& tile)
{
    assert(runningTaskBlock != 0);
    Thread* nextThread = taskBlock.CreateNextThread(tile);

    PrintLine("Thread starting: "
              << runningTaskBlock->task->name
              << " threadIdx=" << nextThread->threadIdx
              << " tileIdx=" << tile.tileIdx);

    tile.core.StartThread(nextThread);
}


/// Is called by Core when it reached the end of it's current instruction stream
void CoreBlock::OnThreadFinished(Thread& finishedThread)
{
    assert(runningTaskBlock != 0);
    assert(runningTaskBlock == finishedThread.taskBlock);

    PrintLine("Thread finished: "
              << runningTaskBlock->task->name
              << " threadIdx=" << finishedThread.threadIdx);

    // Keep the tile to schedule next thread
    Tile* tile = finishedThread.tile;

    // Notify the task block that this thread is finished
    runningTaskBlock->OnThreadFinished(finishedThread);

    if (runningTaskBlock->HasMoreThreads())
    {
        // Schedule next thread
        ScheduleThread(*runningTaskBlock, *tile);
    }
    else if (runningTaskBlock->IsFinished())
    {
        // All blocks of this task have already been scheduled
        processor->OnTaskBlockFinished(*runningTaskBlock);
    }
    else
    {
        // Do nothing: The tile is now idle
        PrintLine("Tile idle: " << tile->tileIdx);
    }
}
