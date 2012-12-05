/// A core block contains multiple tiles that are used to schedule the threads of one TaskBlock
struct CoreBlock
{
    /// The processor to whic this block belongs
    Processor* processor;

    /// All tiles in this block
    Tile* tiles;
    
    /// The index of this core block inside the grid
    int2 blockIdx;
    
    CoreBlock()
    {
        tiles = new Tile[GlobalConfig.CoreBlockSize()];
    }

    ~CoreBlock()
    {
        delete [] tiles;
    }
    

    /// Initializes this CoreBlock
    void InitCoreBlock(params)
    {
        foreach (tile in tiles)
        {
            // TODO: Pass init parameters to each tile
            tile.InitTile(...);
        }
    }
    

    /// The index of the first tile within this core block
    int2 ComputeCoreBlockOrigin()
    {
        return blockIdx * processor->coreBlockSize;
    }


    /// Put the next thread of the given TaskBlock on the given tile
    void ScheduleThread(TaskBlock& taskBlock, Tile& tile)
    {   
        Thread nextThread = taskBlock.CreateNextThread(tile);

#ifdef _VERBOSE
        if (nextThread.threadIdx.Area() % 100)
        {
            PrintLine("Thread starting: " << task.name << " (" << nextThread.threadIdx.x << ", " << nextThread.threadIdx.y << ") on Tile (" << 
                << tile.tileIdx.x << ", " << tile.tileIdx.y << ")";
        }
#endif
        tile.core.StartThread(nextThread);
    }


    /// Is called by Core when it reached the end of it's current instruction stream
    void OnThreadFinished(Thread& thread)
    {
#ifdef _VERBOSE
        if (nextThread.threadIdx.Area() % 100)
        {
            PrintLine("Thread fininshed: " << task.name << " (" << nextThread.threadIdx.x << ", " << nextThread.threadIdx.y << ")");
        }
#endif
        if (thread.taskBlock->HasMoreThreads())
        {
            // Schedule next thread
            ScheduleThread(*thread.taskBlock, *thread.tile);
        }
        else if (thread.taskBlock->IsFinished())
        {
            // All blocks of this task have already been scheduled
            processor->OnTaskBlockFinished(*thread.taskBlock);
        }
        else
        {
            // Do nothing: The tile is now idle
            PrintLine("Tile idle: " << " (" << thread.tile->tileIdx.x << ", " << thread.tile->tileIdx.y << ")");
        }
    }
};
