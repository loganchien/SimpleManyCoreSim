#ifndef CORE_BLOCK_HPP
#define CORE_BLOCK_HPP

#include "SimConfig.hpp"
#include "simutil.hpp"

struct Processor;
struct TaskBlock;
struct Tile;
struct Thread;

/// A core block contains multiple tiles that are used to schedule the threads of one TaskBlock
struct CoreBlock
{
    /// The processor to whic this block belongs
    Processor* processor;

    /// All tiles in this block
    Tile* tiles;

    /// The index of this core block inside the grid
    int2 blockIdx;

    CoreBlock();

    ~CoreBlock();

    /// Initializes this CoreBlock
    void InitCoreBlock()
    {
#if 0
        foreach (tile in tiles)
        {
            // TODO: Pass init parameters to each tile
            tile.InitTile(...);
        }
#endif
    }

    /// The index of the first tile within this core block
    int2 ComputeCoreBlockOrigin() const;

    /// Computes the tile index of the given L2 chunk within this core block
    int2 ComputeTileIndex(int l2ChunkIdx) const
    {
        return ComputeCoreBlockOrigin() + GlobalConfig.ComputeInCoreBlockIdx2(l2ChunkIdx);
    }

    /// Do Z-order lookup (simple conversion from cache entry index to tile)
    int ComputeL2ChunkIndex(const Address& addr) const
    {
        return ComputeL2ChunkIndex(addr.GetL2ChunkIndex());
    }

    /// Computes the L2 chunk index of the given tile index
    int ComputeL2TileChunkIndex(int2 tileIdx) const
    {
        int2 blockOrigin = ComputeCoreBlockOrigin();
        return ComputeL2ChunkIndex(tileIdx - blockOrigin);
    }

    /// Computes the L2 chunk index of the given in-block indexs
    int ComputeL2ChunkIndex(int2 chunkIdx2) const
    {
        // TODO: Domi
    }

    /// Put the next thread of the given TaskBlock on the given tile
    void ScheduleThread(TaskBlock& taskBlock, Tile& tile);

    /// Is called by Core when it reached the end of it's current instruction stream
    void OnThreadFinished(Thread& thread);
};

#endif // CORE_BLOCK_HPP
