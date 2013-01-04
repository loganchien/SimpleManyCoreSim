#ifndef TILE_HPP
#define TILE_HPP

#include "Dimension.hpp"
#include "Router.hpp"

namespace smcsim {

class Core;
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

public:
    Tile();

    void InitTile(CoreBlock* coreBlock, const Dim2& tileIdx);

    /// Whether this tile is at the core's x = 0, y = 0, x = w-1 or y = h-1
    bool IsBoundaryTile();
};

} // end namespace smcsim

#endif // TILE_HPP
