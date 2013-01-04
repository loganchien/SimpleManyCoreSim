#include "Tile.hpp"

#include "Core.hpp"
#include "CoreBlock.hpp"
#include "Dimension.hpp"
#include "Processor.hpp"

using namespace smcsim;

Tile::Tile()
    : coreBlock(0), core(0), mmu(0)
{
}


void Tile::InitTile(CoreBlock* coreBlock_, const Dim2& tileIdx_)
{
    coreBlock = coreBlock_;
    tileIdx = tileIdx_;

    core = new Core(this);
    mmu = &core->mmu;
}


/// Whether this tile is at the core's x = 0, y = 0, x = w-1 or y = h-1
bool Tile::IsBoundaryTile()
{
    Dim2 blockSize = coreBlock->processor->coreBlockSize;
    Dim2 coreOrigin = coreBlock->ComputeCoreBlockOrigin();

    return (tileIdx.x == coreOrigin.x ||
            tileIdx.y == coreOrigin.y ||
            tileIdx.x == coreOrigin.x + blockSize.x - 1 ||
            tileIdx.y == coreOrigin.y + blockSize.y - 1);
}
