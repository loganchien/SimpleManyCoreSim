#include "Thread.hpp"

#include "Dimension.hpp"

using namespace smcsim;

Thread::Thread(TaskBlock* taskBlock_, const Dim2& threadIdx_, Tile *tile_)
    : taskBlock(taskBlock_), threadIdx(threadIdx_), tile(tile_)
{
}
