#ifndef TASK_BLOCK_HPP
#define TASK_BLOCK_HPP

#include "Thread.hpp"

namespace smcsim {

class CoreBlock;
class Task;
class Tile;

/// Every Task is logically partitioned into TaskBlocks.  One TaskBlock has a
/// fixed size of threads that can be scheduled on an assigned CoreBlock.
class TaskBlock
{ 
public:
    /// The task to which this block belongs
    Task* task;

    /// The core block to which this task block assigned
    CoreBlock* assignedBlock;

    /// The blockIdx of this task block
    Dim2 taskBlockIdx;

    /// The id of the first thread that has not been scheduled yet
    Dim2 nextThreadIdx;

    int finishedCount;

public:
    TaskBlock(Task& task, CoreBlock& assignedBlock, const Dim2& taskBlockIdx);

    /// Initializes this TaskBlock
    void InitTaskBlock();

    /// Whether this TaskBlock still has unscheduled threads
    bool HasMoreThreads() const;

    /// Whether all TaskBlocks of this Task have already finished running
    bool IsFinished();

    /// Update the finished thread counter
    void OnThreadFinished(Thread& finishedThread);

    /// Creates the next Thread from this TaskBlock to run on the given tile
    Thread* CreateNextThread(Tile& tile);
};

} // end namespace smcsim

#endif // TASK_BLOCK_HPP
