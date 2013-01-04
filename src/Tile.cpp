#include "Tile.hpp"

#include "Core.hpp"
#include "CoreBlock.hpp"
#include "Dimension.hpp"
#include "Processor.hpp"

using namespace smcsim;

void Tile::InitTile(const Dim2& tileIdx, CoreBlock* coreBlock)
{
    // TODO: Init index, Core etc.
    core = new Core(this);
    mmu = &core->mmu;
}


/// Whether this tile is at the core's x = 0, y = 0, x = w-1 or y = h-1
bool Tile::IsBoundaryTile()
{
    Dim2 blockSize = coreBlock->processor->coreBlockSize;
    Dim2 coreOrigin = coreBlock->ComputeCoreBlockOrigin();
    return tileIdx.x == coreOrigin.x && tileIdx.y == coreOrigin.y &&
        tileIdx.x == coreOrigin.x + blockSize.x - 1 && tileIdx.y == coreOrigin.y + blockSize.y - 1;
}
