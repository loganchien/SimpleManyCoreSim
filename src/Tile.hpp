#ifndef TILE_HPP
#define TILE_HPP

#include "CPU.hpp"
#include "Dimension.hpp"
#include "Router.hpp"

class CoreBlock;
class MMU;

/// A tile consists of a core and a router
class Tile
{
public:
    /// Index of this tile within the core block
    Dim2 tileIdx;

    /// The block to which this tile belongs
    CoreBlock* coreBlock;

    /// The core of this tile (ARMUlator class)
    Core* core;

    /// The MMU of the core (modified ARMUlator class)
    MMU* mmu;

    /// The router of this tile
    Router router;

    /// Amount of normalized ticks passed since initialization
    long long coreTime;

    /// Whether this tile is not currently doing anything
    bool tileIdle;

public:
    void InitTile(Dim2 tileIdx, CoreBlock* coreBlock);

    /// Whether this tile is at the core's x = 0, y = 0, x = w-1 or y = h-1
    bool IsBoundaryTile();
};

#endif // TILE_HPP
