#include "Task.hpp"

#include "TaskBlock.hpp"


TaskBlock Task::CreateNextTaskBlock(CoreBlock& coreBlock)
{
    assert(HasMoreBlocks());
    TaskBlock nextBlock;
    int lastBlockIdx = 0; // FIXME: Definitely incorrect.
    nextBlock.InitTaskBlock(this, lastBlockIdx, coreBlock);

    nextBlockIdx.Inc(blockSize.x);

    return nextBlock;
}
