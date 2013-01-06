#include "Processor.hpp"

#include "Core.hpp"
#include "CoreBlock.hpp"
#include "Debug.hpp"
#include "Dimension.hpp"
#include "TaskBlock.hpp"
#include "Tile.hpp"

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
    while (!batchFinished)
    {
        SimSteps();
    }

    // Done!
    EvaluateStats();
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
void Processor::CollectStats(TaskBlock& taskBlock)
{
    // TODO: Collect stats from all tiles (Core, MMU, Cache)
    taskBlock.task->Stats;
    int coreBlockArea = GlobalConfig.CoreBlockSize().Area();
    Tile* t = taskBlock.assignedBlock->tiles;

    Cache *l1,*l2;
    long long totalL1AccessCount(0),totalL1MissCount(0),avgL1MissRate(0);	// Cache L1 stats
    long long totalL2AccessCount(0),totalL2MissCount(0),avgL2MissRate(0);	// Cache L1 stats
    long long totalInstructions(0),totalLoadInstructions(0);			// Core (CPU) stats
    long long avgSimTime(0), maxSimTime(0);					// MMU stats
    long long totalPacketsReceived(0);						// Router stats


    for(int i=0; i < coreBlockArea; i++){	// iterate over each tile in CoreBlock to get statistics:
        /// Cache and Router statistics:
        avgSimTime+=t[i].mmu.simTime/coreBlockArea; //normalize to get average
        maxSimTime=std::max(maxSimTime,t[i].mmu.simTime);
        l1 = &t[i].mmu.l1;
        totalL1AccessCount+=l1->simAccessCount;	//or better to directly take averages?
        totalL1MissCount+=l1->simMissCount;
        l2 = &t[i].mmu.l2;
        totalL2AccessCount+=l2->simAccessCount;
        totalL2MissCount+=l2->simMissCount;
        totalPacketsReceived+= t[i].router.simTotalPacketsReceived; 

        /// CPU (Core) statistics:
        totalInstructions=t[i].core.simInstructionCount;
        totalLoadInstructions=t[i].core.simLoadInstructionCount;
    }

    // Calculate averages out of totals:
    avgL1MissRate = totalL1MissCount/(totalL1MissCount+totalL1AccessCount);
    avgL2MissRate = totalL2MissCount/(totalL2MissCount+totalL2AccessCount);

	taskBlock.task->Stats.InstructionCount.TotalCount += totalInstructions;
	taskBlock.task->Stats.LoadInstructionCount.TotalCount += totalLoadInstructions;
	taskBlock.task->Stats.L1AccessCount.TotalCount +=totalL1AccessCount;
	taskBlock.task->Stats.L1MissCount.TotalCount +=totalL1MissCount;
	taskBlock.task->Stats.L2AccessCount.TotalCount +=totalL2AccessCount;
	taskBlock.task->Stats.L2MissCount.TotalCount +=totalL2MissCount;
	taskBlock.task->Stats.TotalSimulationTime.TotalCount += maxSimTime;
	taskBlock.task->Stats.AverageSimulationTimeTile.TotalCount += avgSimTime;
	taskBlock.task->Stats.MemAccessTime.TotalCount ; // TODO ADD!
	taskBlock.task->Stats.TotalRouterPackets.TotalCount += totalPacketsReceived;
	taskBlock.task->Stats.TotalThreadCount++;
}


/// Evaluates and/or writes stats to file
void Processor::EvaluateStats()
{
    // TODO: Evaluate and/or write stats to file (for visualization)
	// writeStatsToFile(taskBlock.task->name,avgL1MissRate,avgL2MissRate,avgPacketsReceived,avgInstructions,avgLoadInstructions);
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
    int threadCount = task.blockSize.Area();

    // Put all initial threads on tiles
    for (int i = 0; i < std::min(tileCount, threadCount); ++i)
    {
        Tile& tile = coreBlock.GetTile(
            Dim2::FromLinear(GlobalConfig.CoreBlockSize(), i));
        coreBlock.ScheduleThread(*taskBlock, tile);
    }
}


/// Called by a CoreBlock when it finished executing the given TaskBlock
void Processor::OnTaskBlockFinished(TaskBlock& taskBlock)
{
    PrintLine("TaskBLock finished: " << taskBlock.task->name
              << " taskBlockIdx=" << taskBlock.taskBlockIdx
              << " coreBlockIdx=" << taskBlock.assignedBlock->blockIdx);

    // Collect stats from the functional units of the block
    CollectStats(taskBlock);

    if (taskBlock.task->HasMoreBlocks())
    {
        // Schedule next TaskBlock on the now free core block
        ScheduleTaskBlock(*taskBlock.task, *taskBlock.assignedBlock);
    }
    else
    {
        Task* task = GetNextTask();
        if (task)
        {
            // Schedule next TaskBlock of a different Task on the now free core
            // block
            ScheduleTaskBlock(*task, *taskBlock.assignedBlock);
        }

        if (taskBlock.task->IsFinished())
        {
            // All blocks of this task have already been scheduled
            OnTaskFinished(*taskBlock.task);
        }
    }

    delete &taskBlock;
}


/// Called when the given Task has completed execution of all it's threads
void Processor::OnTaskFinished(Task& task)
{
    PrintLine("Task finished: " << task.name);
	WriteTaskStatsToFile(task.Stats);
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
