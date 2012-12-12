#include "CoreBlock.hpp"

// FIXME: Fix Tile.hpp, Processor.hpp, and Thread.hpp first.
#if 0
#include "Tile.hpp"
#include "SimConfig.hpp"
#include "Processor.hpp"
#include "Thread.hpp"

CoreBlock::CoreBlock()
{
    tiles = new Tile[GlobalConfig.CoreBlockSize()];
}

CoreBlock::~CoreBlock()
{
    delete [] tiles;
}

int2 CoreBlock::ComputeCoreBlockOrigin() const
{
    return blockIdx * processor->coreBlockSize;
}

void CoreBlock::ScheduleThread(TaskBlock& taskBlock, Tile& tile)
{
    Thread nextThread = taskBlock.CreateNextThread(tile);

#ifdef _VERBOSE
    if (nextThread.threadIdx.Area() % 100)
    {
        PrintLine("Thread starting: " << task.name << " (" << nextThread.threadIdx.x << ", " << nextThread.threadIdx.y << ") on Tile (" <<
                  << tile.tileIdx.x << ", " << tile.tileIdx.y << ")");
    }
#endif
    tile.core.StartThread(nextThread);
}

void CoreBlock::OnThreadFinished(Thread& thread)
{
#ifdef _VERBOSE
    if (nextThread.threadIdx.Area() % 100)
    {
        PrintLine("Thread fininshed: " << task.name << " (" << nextThread.threadIdx.x << ", " << nextThread.threadIdx.y << ")");
    }
#endif
    if (thread.taskBlock->HasMoreThreads())
    {
        // Schedule next thread
        ScheduleThread(*thread.taskBlock, *thread.tile);
    }
    else if (thread.taskBlock->IsFinished())
    {
        // All blocks of this task have already been scheduled
        processor->OnTaskBlockFinished(*thread.taskBlock);
    }
    else
    {
        // Do nothing: The tile is now idle
        PrintLine("Tile idle: " << " (" << thread.tile->tileIdx.x << ", " << thread.tile->tileIdx.y << ")");
    }
}
#endif
