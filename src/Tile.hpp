

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
    Router router;
    
    /// Amount of normalized ticks passed since initialization
    long long coreTime;
    
    /// Whether this tile is not currently doing anything
    bool tileIdle;
    
    
    void InitTile(const Index& tileIdx, CoreBlock* coreBlock)
    {
        // TODO: Init index, Core etc.
        
        core = new Core(params);
        mmu = core->mmu;
    }


    /// Whether this tile is at the core's x = 0, y = 0, x = w-1 or y = h-1
    bool IsBoundaryTile()
    {
        int2 blockSize = coreBlock->processor->coreBlockSize;
        int2 coreOrigin = coreBlock->ComputeCoreBlockOrigin();
        return tileIdx.x == coreOrigin.x && tileIdx.y == coreOrigin.y &&
            tileIdx.x == coreOrigin.x + blockSize.x - 1 && tileIdx.y == coreOrigin.y + blockSize.y - 1;
    }
};