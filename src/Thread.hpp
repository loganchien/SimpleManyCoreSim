#ifndef THREAD_HPP
#define THREAD_HPP

#include "Dimension.hpp"

class TaskBlock;
class Tile;

/// A thread is an instruction streams that runs on one Core
class Thread
{
public:
    /// The tile on which this Thread runs
    Tile* tile;

    /// The TaskBlock to which this Thread belongs
    TaskBlock* taskBlock;

    /// The index of this thread within it's TaskBlock
    Dim2 threadIdx;

    /// This thread's instruction stream
    Code code;

    void InitThread();
};

#endif // THREAD_HPP