#include "Tile.hpp"

#include <stdlib.h>

bool Tile::IsBoundaryTile()
{
#if 0
    int2 blockSize = coreBlock->processor->coreBlockSize;
    int2 coreOrigin = coreBlock->ComputeCoreBlockOrigin();
    return tileIdx.x == coreOrigin.x && tileIdx.y == coreOrigin.y &&
        tileIdx.x == coreOrigin.x + blockSize.x - 1 && tileIdx.y == coreOrigin.y + blockSize.y - 1;
#else
    assert(0 && "not implemented");
    abort();
#endif
}
