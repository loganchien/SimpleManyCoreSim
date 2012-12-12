#include "simutil.hpp"

#include <vector>
#include <stdint.h>

struct CoreBlock;
struct TaskBlock;

/// One stat value
struct TaskStat
{
    void SetStat(double totalCount, int threadCount)
    {
        TotalCount = totalCount;
        AvgCount = TotalCount / threadCount;
    }

    double TotalCount;

    double AvgCount;
};

/// Stats for one task
struct TaskStats
{
    /// # of executed threads
    long long TotalThreadCount;

    /// # of instructions per thread
    TaskStat InstructionCount;

    /// # of load instructions
    TaskStat LoadInstructionCount;

    /// Simulation time spent on memory access
    TaskStat MemAccessTime;

    /// # of L1 accesses & misses
    TaskStat L1AccessCount, L1MissCount;

    /// # of L2 accesses & misses
    TaskStat L2AccessCount, L2MissCount;
};


/// A Task has code and some meta information
struct Task
{
    /// Name
    std::string name;

    /// Code to be executed
    std::vector<uint8_t> code;

    /// Total and per-block size
    int2 taskSize, blockSize;

    /// The index of the TaskBlock that is to be scheduled next
    int2 nextBlockIdx;

    /// The amount of blocks that have already finished executing 
    int finishedCount;

    /// All relevant stats, recorded during execution
    TaskStats Stats;

    /// Whether this Task still has unscheduled TaskBlocks
    bool HasMoreBlocks() const
    {
        return nextBlockIdx.Area() <= taskSize.Area();
    }

    /// Whether all TaskBlocks of this Task have already finished running
    bool IsFinished()
    {
        return finishedCount == taskSize.Area();
    }

    /// Creates the next TaskBlock in this task
    TaskBlock CreateNextTaskBlock(CoreBlock& coreBlock);
};
