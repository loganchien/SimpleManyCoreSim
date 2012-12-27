#include "TaskBlock.hpp"

TaskBlock()
{
}

/// Initializes this TaskBlock
void TaskBlock::InitTaskBlock()
{
    // TODO: Init TaskBlock

    assignedBlock->taskBlock = this;
}

/// Instruments the Task code for this block (i.e. insert block-id, thread-id etc into special placeholders within the code)
Program *TaskBlock::GetInjectedCode(int2 threadId)
{
    // TODO: Replace placeholders in constant segment with thread id information
    return NULL;
}


/// Whether this TaskBlock still has unscheduled threads
bool TaskBlock::HasMoreThreads() const
{
    return nextThreadId.Area() <= task->blockSize.Area();
}

/// Whether all TaskBlocks of this Task have already finished running
bool TaskBlock::IsFinished()
{
    return finishedCount == task->blockSize.Area();
}

/// Creates the next Thread from this TaskBlock to run on the given tile
Thread TaskBlock::CreateNextThread(Tile& tile)
{
    assert(HasMoreThreads());

    Thread nextThread;
    nextThread.InitThread(this, nextThreadId, tile, GetInjectedCode(nextThreadId));

    nextThreadId.Inc(1);

    return nextThread;
}
