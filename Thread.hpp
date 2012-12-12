#ifndef THREAD_HPP
#define THREAD_HPP

#include "simutil.hpp"

#include <vector>
#include <stdint.h>

struct Tile;
struct TaskBlock;

/// A thread is an instruction streams that runs on one Core
struct Thread
{
    /// The tile on which this Thread runs
    Tile* tile;

    /// The TaskBlock to which this Thread belongs
    TaskBlock* taskBlock;

    /// The index of this thread within it's TaskBlock
    int2 threadIdx;

    /// This thread's instruction stream
    std::vector<uint8_t> code;

    void InitThread(TaskBlock*, int2, Tile&, const std::vector<uint8_t>&)
    {
        // TODO: Init thread
    }
};

#endif // THREAD_HPP
