#ifndef TASK_HPP
#define TASK_HPP

#include "Dimension.hpp"

#include <string>
#include <stdint.h>

namespace smcsim {

class CoreBlock;
class TaskBlock;
class Program;

/// One stat value
class TaskStat
{
public:
    void SetStat(double totalCount, int threadCount)
    {
        TotalCount = totalCount;
        AvgCount = TotalCount / threadCount;
    }

    double TotalCount;

    double AvgCount;
};

/// Stats for one task
class TaskStats
{
public:
    /// # of executed threads
    long long TotalThreadCount;

    /// # of instructions per thread
    TaskStat InstructionCount;

    /// # of load instructions
    TaskStat LoadInstructionCount;

	/// Simulation time (total) of a task
	TaskStat TotalSimulationTime;

	/// Average simulation time of each core in the coreblock(busy period) -> find avg idle time from (max - avg)?
	TaskStat AverageSimulationTimeTile;

    /// Simulation time spent on memory access
    TaskStat MemAccessTime;

	/// Total number of packets received in router
	TaskStat TotalRouterPackets;

    /// # of L1 accesses & misses
    TaskStat L1AccessCount, L1MissCount;

    /// # of L2 accesses & misses
    TaskStat L2AccessCount, L2MissCount;
};


/// A Task has code and some meta information
class Task
{
public:
    /// Name
    std::string name;

    /// Code to be executed
    std::string elfFilePath;

    /// The address of predefined threadIdx
    uint32_t threadIdxAddr;

    /// The address of predefined threadDim
    uint32_t threadDimAddr;

    /// The address of predefined blockIdx
    uint32_t blockIdxAddr;

    /// The address of predefined blockDim
    uint32_t blockDimAddr;

    /// Thread dimension specified by the programmer.
    Dim2 threadDim;

    /// Task block dimension specified by the programmer.
    Dim2 blockDim;

    /// The index of the TaskBlock that is to be scheduled next
    Dim2 nextBlockIdx;

    /// The amount of blocks that have already finished executing
    int finishedCount;

    /// All relevant stats, recorded during execution
    TaskStats Stats;

    /// Whether this Task still has unscheduled TaskBlocks
    bool HasMoreBlocks() const;

    /// Whether all TaskBlocks of this Task have already finished running
    bool IsFinished();

public:
    /// Creates the next TaskBlock in this task
    TaskBlock* CreateNextTaskBlock(CoreBlock& coreBlock);

    /// Load the task configuration from the configuration file
    static Task* Create(const std::string& path);

	/// Write stats of task to CSV file
	void WriteTaskStatsToFile();

private:
    Task(const std::string& name,
         const std::string& elfFilePath,
         uint32_t threadIdxAddr, uint32_t threadDimAddr,
         uint32_t blockIdxAddr, uint32_t blockDimAddr,
         const Dim2& threadDim, const Dim2& blockDim);
};

} // end namespace smcsim

#endif // TASK_HPP
