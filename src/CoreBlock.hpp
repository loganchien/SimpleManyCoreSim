#ifndef CORE_BLOCK_HPP
#define CORE_BLOCK_HPP

#include "Dimension.hpp"

class Address;
class Processor;
class TaskBlock;
class Thread;
class Tile;

/// A core block contains multiple tiles that are used to schedule the threads
/// of one TaskBlock
class CoreBlock
{
public:
    /// The index of this core block inside the grid
    Dim2 blockIdx;

    /// The processor to which this block belongs
    Processor* processor;

    /// All tiles in this block
    Tile* tiles;

    /// Running task block
    TaskBlock* runningTaskBlock;

public:
    CoreBlock();

    ~CoreBlock();

    /// Initializes this CoreBlock
    void InitCoreBlock(Processor* processor, const Dim2& blockIdx);

    /// The index of the first tile within this core block
    Dim2 ComputeCoreBlockOrigin() const;


    /// Computes the global 2D index of the given block-local 1D index
    Dim2 ComputeTileIndex(int inCoreID) const;

    /// Computes the L2 chunk index of the given memory address
    int ComputeL2ChunkID(const Address& addr) const;


    /// Computes the L2 chunk index of the given global tile index
    int ComputeL2TileChunkID(const Dim2& globalIdx) const;


    /// Computes the L2 chunk ID of the given block-local 2D index
    int ComputeL2ChunkID(const Dim2& inCoreBlockIdx2) const;


    /// Put the next thread of the given TaskBlock on the given tile
    void ScheduleThread(TaskBlock& taskBlock, Tile& tile);


    /// Is called by Core when it reached the end of it's current instruction
    /// stream
    void OnThreadFinished(Thread& thread);
};

#endif // CORE_BLOCK_HPP
