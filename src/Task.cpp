#include "Task.hpp"

#include "TaskBlock.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

#include <iomanip>
#include <sstream>

#include <assert.h>
#include <stdint.h>

using namespace boost;
using namespace boost::property_tree;
using namespace boost::property_tree::ini_parser;
using namespace std;


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
           const Dim2& taskSize_, const Dim2& blockSize_)
    : name(name_), elfFilePath(elfFilePath_),
      threadIdxAddr(threadIdxAddr_), threadDimAddr(threadDimAddr_),
      blockIdxAddr(blockIdxAddr_), blockDimAddr(blockDimAddr_),
      taskSize(taskSize_), blockSize(blockSize_), finishedCount(0)
{
}

Task* Task::Create(const std::string& path)
{
    ptree pt;
    read_ini(path, pt);

    string name(pt.get<string>("task.executable")); // TODO: Change to name
    string executable(pt.get<string>("task.executable"));

    uint32_t threadIdxAddr(GetHexValue(pt.get<string>("task.thread_idx_addr")));
    uint32_t threadDimAddr(GetHexValue(pt.get<string>("task.thread_dim_addr")));
    uint32_t blockIdxAddr(GetHexValue(pt.get<string>("task.block_idx_addr")));
    uint32_t blockDimAddr(GetHexValue(pt.get<string>("task.block_dim_addr")));

    int threadWidth(pt.get<int>("task.thread_width"));
    int threadHeight(pt.get<int>("task.thread_height"));
    int blockWidth(pt.get<int>("task.block_width"));
    int blockHeight(pt.get<int>("task.block_height"));

    return new Task(name, executable,
                    threadIdxAddr, threadDimAddr, blockIdxAddr, blockDimAddr,
                    Dim2(threadHeight, threadWidth),
                    Dim2(blockHeight, blockWidth));
}

/// Whether this Task still has unscheduled TaskBlocks
bool Task::HasMoreBlocks() const
{
    return nextBlockIdx.Area() <= taskSize.Area();
}

/// Whether all TaskBlocks of this Task have already finished running
bool Task::IsFinished()
{
    return finishedCount == taskSize.Area();
}

/// Creates the next TaskBlock in this task
TaskBlock Task::CreateNextTaskBlock(CoreBlock& coreBlock)
{
    assert(HasMoreBlocks());
    TaskBlock nextBlock;
    nextBlock.InitTaskBlock(); // TODO: Initialize the task block correctly.

    nextBlockIdx.Inc(blockSize.x);

    return nextBlock;
}
