#include "Task.hpp"

#include "Debug.hpp"
#include "TaskBlock.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem/path.hpp>

#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdint.h>

using namespace smcsim;
using namespace std;

using boost::filesystem::path;
using boost::property_tree::ptree;
using boost::property_tree::ini_parser::read_ini;

namespace {

uint32_t GetHexValue(const string& str)
{
    uint32_t result = 0;
    stringstream ss(str);
    ss >> hex >> result;
    return result;
}

} // end anonymous namespace


Task::Task(const std::string& name_,
           const std::string& elfFilePath_,
           uint32_t threadIdxAddr_, uint32_t threadDimAddr_,
           uint32_t blockIdxAddr_, uint32_t blockDimAddr_,
           const Dim2& threadDim_, const Dim2& blockDim_)
    : name(name_), elfFilePath(elfFilePath_),
      threadIdxAddr(threadIdxAddr_), threadDimAddr(threadDimAddr_),
      blockIdxAddr(blockIdxAddr_), blockDimAddr(blockDimAddr_),
      threadDim(threadDim_), blockDim(blockDim_), finishedCount(0)
{
}

Task* Task::Create(const std::string& taskConfigPath)
{
    ptree pt;
    read_ini(taskConfigPath, pt);

    string name(pt.get<string>("task.executable")); // TODO: Change to name

    path executablePath(taskConfigPath);
    executablePath.remove_filename() /= pt.get<string>("task.executable");

    uint32_t threadIdxAddr(GetHexValue(pt.get<string>("task.thread_idx_addr")));
    uint32_t threadDimAddr(GetHexValue(pt.get<string>("task.thread_dim_addr")));
    uint32_t blockIdxAddr(GetHexValue(pt.get<string>("task.block_idx_addr")));
    uint32_t blockDimAddr(GetHexValue(pt.get<string>("task.block_dim_addr")));

    int threadWidth(pt.get<int>("task.thread_width"));
    int threadHeight(pt.get<int>("task.thread_height"));
    int blockWidth(pt.get<int>("task.block_width"));
    int blockHeight(pt.get<int>("task.block_height"));

    Dim2 threadDim(threadHeight, threadWidth);
    Dim2 blockDim(blockHeight, blockWidth);

    PrintLine("TASK: " << name << "  threadDim=" << threadDim
                               << "  blockDim=" << blockDim);

    return new Task(name, executablePath.string(),
                    threadIdxAddr, threadDimAddr, blockIdxAddr, blockDimAddr,
                    threadDim, blockDim);
}

/// Whether this Task still has unscheduled TaskBlocks
bool Task::HasMoreBlocks() const
{
    return nextBlockIdx.ToLinear(blockDim) < blockDim.Area();
}

/// Whether all TaskBlocks of this Task have already finished running
bool Task::IsFinished()
{
    return finishedCount == blockDim.Area();
}

/// Creates the next TaskBlock in this task
TaskBlock* Task::CreateNextTaskBlock(CoreBlock& coreBlock)
{
    assert(HasMoreBlocks());
    TaskBlock* nextBlock = new TaskBlock(*this, coreBlock, nextBlockIdx);
    nextBlockIdx.Inc(blockDim.x);
    return nextBlock;
}

void Task::WriteTaskStatsToFile(){
	ofstream statFile;
	char path[] = "//results//taskstats.txt";
	statFile.open (path,ios::out | ios::app);
	statFile << name << "," << Stats.TotalThreadCount 
		<< "," << Stats.InstructionCount.TotalCount
	    << "," << Stats.LoadInstructionCount.TotalCount
		<< "," << Stats.TotalSimulationTime.TotalCount
		<< "," << Stats.AverageSimulationTimeTile.TotalCount
		<< "," << Stats.MemAccessTime.TotalCount
		<< "," << Stats.TotalRouterPackets.TotalCount
		<< "," << Stats.L1AccessCount.TotalCount 
		<< "," << Stats.L1MissCount.TotalCount
		<< "," << Stats.L2AccessCount.TotalCount
		<< "," << Stats.L2MissCount.TotalCount
		<< endl ;
	
	statFile.close();
}
