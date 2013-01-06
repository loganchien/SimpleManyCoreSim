#include "TaskBlock.hpp"

#include "CoreBlock.hpp"
#include "Debug.hpp"
#include "Dimension.hpp"
#include "Task.hpp"

#include <assert.h>

using namespace smcsim;

TaskBlock::TaskBlock(Task& task_, CoreBlock& assignedBlock_,
                     const Dim2& taskBlockIdx_)
    : task(&task_), assignedBlock(&assignedBlock_),
      taskBlockIdx(taskBlockIdx_),finishedCount(0)
{
}

/// Whether this TaskBlock still has unscheduled threads
bool TaskBlock::HasMoreThreads() const
{
    return nextThreadIdx.Area() <= task->blockSize.Area();
}

/// Whether all TaskBlocks of this Task have already finished running
bool TaskBlock::IsFinished()
{
    return finishedCount == task->blockSize.Area();
}

/// Creates the next Thread from this TaskBlock to run on the given tile
Thread* TaskBlock::CreateNextThread(Tile& tile)
{
    assert(HasMoreThreads());

    Thread *nextThread = new Thread(this, nextThreadIdx, &tile);

    nextThreadIdx.Inc(task->taskSize.x);

    return nextThread;
}
