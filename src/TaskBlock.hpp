#ifndef TASK_BLOCK_HPP
#define TASK_BLOCK_HPP

#include "Thread.hpp"

class CoreBlock;
class Task;
class Tile;

/**
 * Every Task is logically partitioned into TaskBlocks.
 * One TaskBlock has a fixed size of threads that can be scheduled on an assigned CoreBlock.
 */
class TaskBlock
{
public:
    /// The task to which this block belongs
    Task* task;
    Dim2 taskBlockIdx;
    CoreBlock* assignedBlock;

    /// The id of the first thread that has not been scheduled yet
    Dim2 nextThreadIdx;

    int finishedCount;

public:
    TaskBlock();

    /// Initializes this TaskBlock
    void InitTaskBlock();

    /// Instruments the Task code for this block (i.e. insert block-id, thread-id etc into special placeholders within the code)
    Program *GetInjectedCode(const Dim2& threadIdx);

    /// Whether this TaskBlock still has unscheduled threads
    bool HasMoreThreads() const;

    /// Whether all TaskBlocks of this Task have already finished running
    bool IsFinished();

    /// Creates the next Thread from this TaskBlock to run on the given tile
    Thread CreateNextThread(Tile& tile);
};

#endif // TASK_BLOCK_HPP
