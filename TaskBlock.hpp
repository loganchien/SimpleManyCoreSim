

struct Tile;

/**
 * Every Task is logically partitioned into TaskBlocks. 
 * One TaskBlock has a fixed size of threads that can be scheduled on an assigned CoreBlock.
 */
struct TaskBlock
{
    /// The task to which this block belongs
    Task* task;
    int2 taskBlockIdx;
    CoreBlock* assignedBlock;

    /// The id of the first thread that has not been scheduled yet
    int2 nextThreadId;

    TaskBlock() {}
    
    /// Initializes this TaskBlock
    void InitTaskBlock(params)
    {
        // TODO: Init TaskBlock

        assignedBlock->taskBlock = this;
    }

    /// Instruments the Task code for this block (i.e. insert block-id, thread-id etc into special placeholders within the code)
    Code GetInjectedCode(int2 threadId)
    {
        // copy original code
        Code origCode = task->code;
        
        // TODO: Replace placeholders in constant segment with thread id information

        return code;
    }
    

    /// Whether this TaskBlock still has unscheduled threads
    bool HasMoreThreads() const
    {
        return nextThreadId.Area() <= task->blockSize.Area();
    }

    /// Whether all TaskBlocks of this Task have already finished running
    bool IsFinished()
    {
        return finishedCount == task->blockSize.Area();
    }

    /// Creates the next Thread from this TaskBlock to run on the given tile
    Thread CreateNextThread(Tile& tile)
    {
        assert(HasMoreThreads());

        Thread nextThread;
        nextThread.InitTaskBlock(this, lastBlockIdx, coreBlock);
        
        nextThreadId.Inc(task->blockSize.x);

        return nextBlock;
    }
};