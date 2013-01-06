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

#include "assert.h"

using namespace smcsim;

CoreBlock::CoreBlock()
{
    tiles = new Tile[GlobalConfig.CoreBlockSize().Area()];
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

    for (int i = 0; i < GlobalConfig.CoreBlockSize().Area(); ++i)
    {
        tiles[i].InitTile(this,
                          Dim2::FromLinear(GlobalConfig.CoreBlockSize(), i));
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
Tile& CoreBlock::GetTile(const Dim2& tileIdx)
{
    return tiles[Dim2::ToLinear(GlobalConfig.CoreBlockSize(), tileIdx)];
}


/// Get the Tile by tileIdx
const Tile& CoreBlock::GetTile(const Dim2& tileIdx) const
{
    return tiles[Dim2::ToLinear(GlobalConfig.CoreBlockSize(), tileIdx)];
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
void CoreBlock::OnThreadFinished(Thread& thread)
{
    assert(runningTaskBlock != 0);
    PrintLine("Thread finished: "
              << runningTaskBlock->task->name
              << " threadIdx=" << thread.threadIdx);

    if (thread.taskBlock->HasMoreThreads())
    {
        // Schedule next thread
        ScheduleThread(*thread.taskBlock, *thread.tile);
        delete &thread;
    }
    else if (thread.taskBlock->IsFinished())
    {
        // All blocks of this task have already been scheduled
        processor->OnTaskBlockFinished(*thread.taskBlock);
        delete &thread;
    }
    else
    {
        // Do nothing: The tile is now idle
        PrintLine("Tile idle: " << thread.tile->tileIdx);
    }
}
