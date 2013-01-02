#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include "Dimension.hpp"
#include "GlobalMemoryController.hpp"
#include "Task.hpp"

#include <vector>

class CoreBlock;
class TaskBlock;
class Tile;

/// The Processor contains and does it all
class Processor
{
public:
    /// The amount and size of blocks on this processor
    Dim2 coreGridSize, coreBlockSize;

    /// All blocks available on this processor
    CoreBlock* blocks;

    /// The Global Memory Controller is responsible for supplying data that is
    /// not in any cache
    GlobalMemoryController gMemController;

    /// The current batch of tasks to run
    std::vector<Task> tasks;

    /// Whether the current batch of tasks has finished running
    bool batchFinished;

    /// The number of the current batch
    int batchNum;

public:
    Processor();

    ~Processor();

    // #################### Simulation ########################################

    /// The entry point of any simulation: Run a batch of tasks
    void StartBatch(const std::vector<Task>& tasks);

    /// Take a few simulation steps
    void SimSteps();

    /// Collect stats from the functional units of the block
    void CollectStats(TaskBlock& taskBlock);

    /// Evaluates and/or writes stats to file
    void EvaluateStats();

    // #################### Task management ###################################

    /// Gets the next task that still has unscheduled blocks or null, if there
    /// is none
    Task* GetNextTask();

    /// Put the next TaskBlock of the given Task on the given CoreBlock
    void ScheduleTaskBlock(Task& task, CoreBlock& coreBlock);

    /// Called by a CoreBlock when it finished executing the given TaskBlock
    void OnTaskBlockFinished(TaskBlock& taskBlock);

    /// Called when the given Task has completed execution of all it's threads
    void OnTaskFinished(Task& task);

    /// Called when all tasks have been worked through
    void OnBatchFinished();

    // ########################################################################
    Tile* GetTile(const Dim2& tileIdx);
};

#endif // PROCESSOR_HPP
