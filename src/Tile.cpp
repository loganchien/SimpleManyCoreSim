#include "Tile.hpp"

#include "Core.hpp"
#include "CoreBlock.hpp"
#include "Debug.hpp"
#include "Dimension.hpp"
#include "Processor.hpp"

using namespace smcsim;

Tile::Tile()
    : coreBlock(0), core(this, &mmu)
{
}


void Tile::InitTile(CoreBlock* coreBlock_, const Dim2& tileIdx_)
{
    PrintLine("InitTile: " << tileIdx_);
    coreBlock = coreBlock_;
    tileIdx = tileIdx_;
    PrintLine("InitMMU: " << tileIdx_);
    mmu.InitMMU(this);
}


/// Whether this tile is at the core's x = 0, y = 0, x = w-1 or y = h-1
bool Tile::IsBoundaryTile()
{
    Dim2 blockSize = coreBlock->processor->coreBlockSize;
    Dim2 coreOrigin = coreBlock->ComputeCoreBlockOrigin();

    return (tileIdx.x == 0 ||
            tileIdx.y == 0 ||
            tileIdx.x == GlobalConfig.TotalCoreLength() - 1 ||
            tileIdx.y == GlobalConfig.TotalCoreLength() - 1);
}

Dim2 Tile::ComputeLocalIndex(){
    int x = tileIdx.x % GlobalConfig.CoreBlockLen;
    int y = tileIdx.y % GlobalConfig.CoreBlockLen;
    return Dim2(y,x);
}

int Tile::GetGlobalLinearIndex() const
{
    return tileIdx.y * GlobalConfig.TotalCoreLength() + tileIdx.x;
}
