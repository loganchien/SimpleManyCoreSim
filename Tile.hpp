#ifndef TILE_HPP
#define TILE_HPP

#include "simutil.hpp"

struct Core;
struct CoreBlock;
struct MMU;
struct Router;

/// A tile consists of a core and a router
struct Tile
{
    /// Index of this tile within the core block
    int2 tileIdx;

    /// The block to which this tile belongs
    CoreBlock* coreBlock;

    /// The core of this tile (ARMUlator class)
    Core* core;

    /// The MMU of the core (modified ARMUlator class)
    MMU* mmu;

    /// The router of this tile
    Router* router;

    /// Amount of normalized ticks passed since initialization
    long long coreTime;

    /// Whether this tile is not currently doing anything
    bool tileIdle;


    void InitTile(const int2& tileIdx, CoreBlock* coreBlock)
    {
        // TODO: Init index, Core etc.
#if 0
        // FIXME: Not implemented
        core = new Core(params);
        mmu = core->mmu;
#endif
    }

    /// Whether this tile is at the core's x = 0, y = 0, x = w-1 or y = h-1
    bool IsBoundaryTile();
};

#endif // TILE_HPP
