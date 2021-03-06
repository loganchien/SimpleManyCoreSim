#include "Processor.hpp"

#include "Core.hpp"
#include "CoreBlock.hpp"
#include "Debug.hpp"
#include "Dimension.hpp"
#include "TaskBlock.hpp"
#include "Tile.hpp"

#include <iomanip>
#include <iostream>
#include <algorithm>

#include <assert.h>

using namespace smcsim;
using namespace std;

Processor::Processor()
{
    coreGridSize = GlobalConfig.CoreGridSize();
    coreBlockSize = GlobalConfig.CoreBlockSize();

    batchNum = 0;
    blocks = new CoreBlock[coreGridSize.Area()];
    batchFinished = true;
    gMemController.InitGMemController(this);

    for (int i = 0; i < coreGridSize.Area(); ++i)
    {
        blocks[i].InitCoreBlock(this, Dim2::FromLinear(coreGridSize, i));
    }
}


Processor::~Processor()
{
    delete [] blocks;
}


// #################### Simulation ############################################

/// The entry point of any simulation: Run a batch of tasks
void Processor::StartBatch(const std::vector<Task>& tasks)
{
    assert(tasks.size() > 0);

    ++batchNum;
    batchFinished = false;

    PrintLine("Batch starting: " << batchNum);

    this->tasks = tasks;

    // Assign initial task blocks in round-robin fashion
    for (int i = 0; i < coreGridSize.Area(); ++i)
    {
        Task* task = GetNextTask();
        if (!task) break;

        ScheduleTaskBlock(*task, blocks[i]);
    }

    // Start running the simulation
    long long clock = 0;
    while (!batchFinished)
    {
        SimSteps();
        clock++;
    }

    // Done!
    EvaluateStats();

    std::cout << "Clock used: " << clock << std::endl;
}


/// Take a few simulation steps
void Processor::SimSteps()
{
    // Simulate one GMem step
    //gMemController.DispatchNext();

    // Simulate one step for each tile
    for (int bi = 0; bi < coreGridSize.Area(); ++bi)
    {
        // One iteration per block
        CoreBlock& block = blocks[bi];

        for (int i = 0; i < coreBlockSize.Area(); ++i)
        {
            // One iteration per tile
            Tile& tile = block.tiles[i];
            tile.core.DispatchNext();
            //tile.router.DispatchNext();
        }
    }
}


/// Collect stats from the functional units of the block
void Processor::CollectStats(TaskBlock& taskBlock,
                             long long& totalL1AccessCount,
                             long long& totalL1MissCount,
                             long long& totalL2AccessCount,
                             long long& totalL2MissCount)
{
    long long totalInstructions(0),totalLoadInstructions(0);
    long long avgSimTime(0), totalSimTime(0);
    long long totalPacketsReceived(0);

    // Collect stats from all tiles (Core, MMU, Cache)
    int coreBlockArea = GlobalConfig.CoreBlockSize().Area();
    Tile* t = taskBlock.assignedBlock->tiles;
    for(int i = 0; i < coreBlockArea; i++)
    {
        /// Cache and Router statistics:
        avgSimTime += t[i].mmu.simTime / coreBlockArea;
        totalSimTime += t[i].mmu.simTime;

        Cache* l1 = &t[i].mmu.l1;
        totalL1AccessCount += l1->simAccessCount;
        totalL1MissCount += l1->simMissCount;

        Cache* l2 = &t[i].mmu.l2;
        totalL2AccessCount += l2->simAccessCount;
        totalL2MissCount += l2->simMissCount;
        totalPacketsReceived += t[i].router.simTotalPacketsReceived;

        /// CPU (Core) statistics:
        totalInstructions = t[i].core.simInstructionCount;
        totalLoadInstructions = t[i].core.simLoadInstructionCount;
    }

    taskBlock.task->Stats.InstructionCount.TotalCount += totalInstructions;
    taskBlock.task->Stats.LoadInstructionCount.TotalCount += totalLoadInstructions;
    taskBlock.task->Stats.L1AccessCount.TotalCount +=totalL1AccessCount;
    taskBlock.task->Stats.L1MissCount.TotalCount +=totalL1MissCount;
    taskBlock.task->Stats.L2AccessCount.TotalCount +=totalL2AccessCount;
    taskBlock.task->Stats.L2MissCount.TotalCount +=totalL2MissCount;
    taskBlock.task->Stats.TotalSimulationTime.TotalCount += totalSimTime;
    taskBlock.task->Stats.AverageSimulationTimeTile.TotalCount += avgSimTime;
    //taskBlock.task->Stats.MemAccessTime.TotalCount ; // TODO ADD!
    taskBlock.task->Stats.TotalRouterPackets.TotalCount += totalPacketsReceived;
    taskBlock.task->Stats.TotalThreadCount++;
}


/// Evaluates and/or writes stats to file
void Processor::EvaluateStats()
{
    // Evaluate and/or write stats to file (for visualization)

}

// #################### Task management #######################################

/// Gets the next task that still has unscheduled blocks or null, if there is
/// none
Task* Processor::GetNextTask()
{
    int t = 0;
    for (std::vector<Task>::iterator it = tasks.begin(); it != tasks.end(); ++it)
    {
        if (it->HasMoreBlocks())
        {
            if (gMemController.task != &*it)
            {
                gMemController.LoadExecutable(&*it);
            }
            return &*it;
        }
    }
    return NULL;
}


/// Put the next TaskBlock of the given Task on the given CoreBlock
void Processor::ScheduleTaskBlock(Task& task, CoreBlock& coreBlock)
{
    // Create TaskBlock
    TaskBlock* taskBlock = task.CreateNextTaskBlock(coreBlock);

    // Assign this TaskBlock to this CoreBlock
    coreBlock.runningTaskBlock = taskBlock;

    PrintLine("TaskBlock starting: " << task.name
              << " taskBlockIdx=" << taskBlock->taskBlockIdx
              << " coreBlockIdx=" << coreBlock.blockIdx);

    int tileCount = GlobalConfig.CoreBlockSize().Area();
    int threadCount = task.threadDim.Area();

    // Reset the cache stats
    for (int i = 0; i < tileCount; ++i)
    {
        Tile& tile = coreBlock.tiles[i];
        tile.mmu.l1.simAccessCount = 0;
        tile.mmu.l1.simMissCount = 0;
        tile.mmu.l2.simAccessCount = 0;
        tile.mmu.l2.simMissCount = 0;
    }

    // Put all initial threads on tiles
    for (int i = 0; i < std::min(tileCount, threadCount); ++i)
    {
        Tile& tile = coreBlock.tiles[i];
        coreBlock.ScheduleThread(*taskBlock, tile);
    }
}


/// Called by a CoreBlock when it finished executing the given TaskBlock
void Processor::OnTaskBlockFinished(TaskBlock& taskBlock)
{
    Task* task = taskBlock.task;
    CoreBlock* coreBlock = taskBlock.assignedBlock;

    assert(task);
    assert(coreBlock);

    PrintLine("TaskBLock finished: " << task->name
              << " taskBlockIdx=" << taskBlock.taskBlockIdx
              << " coreBlockIdx=" << coreBlock->blockIdx);

    // Collect stats from the functional units of the block
    long long totalL1AccessCount(0), totalL1MissCount(0);
    long long totalL2AccessCount(0),totalL2MissCount(0);
    CollectStats(taskBlock,
                 totalL1AccessCount, totalL1MissCount,
                 totalL2AccessCount, totalL2MissCount);

    long long totalL1HitCount = totalL1AccessCount - totalL1MissCount;
    long long totalL2HitCount = totalL2AccessCount - totalL2MissCount;

    std::cout << "  TB " << std::setw(8) << taskBlock.taskBlockIdx << ":";
    std::cout << " | L1 hit:" << std::setw(8) << totalL1HitCount;
    std::cout << " miss:" << std::setw(8) << totalL1MissCount;
    std::cout << " | L2 hit:" << std::setw(8) << totalL2HitCount;
    std::cout << " miss:" << std::setw(8) << totalL2MissCount;
    std::cout << std::endl;

    // Update the finished task counter
    task->OnTaskBlockFinished(taskBlock);

    if (task->HasMoreBlocks())
    {
        // Schedule next TaskBlock on the now free core block
        ScheduleTaskBlock(*task, *coreBlock);
    }
    else
    {
        if (task->IsFinished())
        {
            // All blocks of this task have already been scheduled
            OnTaskFinished(*task);
        }

        // Schedule a new TaskBlock of the other task to coreBlock.
        task = GetNextTask();
        if (task)
        {
            // Schedule next TaskBlock of a different Task on the now free core
            // block
            ScheduleTaskBlock(*task, *coreBlock);
        }
    }
}


/// Called when the given Task has completed execution of all it's threads
void Processor::OnTaskFinished(Task& task)
{
    PrintLine("Task finished: " << task.name);

    task.WriteTaskStatsToFile();

    if (!GetNextTask())
    {
        // The batch has finished
        OnBatchFinished();
    }
}


/// Called when all tasks have been worked through
void Processor::OnBatchFinished()
{
    PrintLine("Batch finished: " << batchNum);
    batchFinished = true;
}


/// Get the tile
Tile* Processor::GetTile(const Dim2& tileIdx)
{
    div_t dx = div(tileIdx.x, coreBlockSize.x);
    div_t dy = div(tileIdx.y, coreBlockSize.y);
    int blockIdx = dx.quot + dy.quot * coreGridSize.x;
    CoreBlock& cb = blocks[blockIdx];
    return &cb.tiles[dx.rem + dy.rem * coreBlockSize.x];
}
