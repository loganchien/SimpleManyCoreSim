
/// This is an ARMulator class
struct CPU
{
    /// The tile to which this CPU belongs
    Tile* tile;

    /// The currently running thread
    Thread* currentThread;
    
    /// Whether this Core is currently waiting for off-chip data
    bool isStalling;

    /// The total amount of instructions
    long long simInstructionCount, simLoadInstructionCount;

    // ...

    CPU(Tile* tile)
    {
        this->tile = tile;
        
        mmu.InitMMU(thread->tile);
    }

    void StartThread(Thread* thread)
    {
        currentThread = thread;
        
        mmu.ResetMMU();
        
        // TODO: Put thread.Code into I-Cache
    }


    /// Dispatches and, if possible, executes one simulated instruction. Returns false, if there are no more instruction to execute (i.e. EOF reached).
    bool DispatchNext()
    {
        assert(!IsStalling());
        
        ++simInstructionCount;
        // TODO: Only run/start the next instruction

        // TODO: Call DispatchLoad on load

        // TODO: Call if the current instruction was the last instruction: tile->coreBlock->OnThreadFinished(*currentThread);
    }


    /// Forwards a load instruction to the MMU
    void DispatchLoad(int addrWord)
    {
        simLoadInstructionCount++;
        Address addr(addrWord);
        mmu.LoadWord(addr);
    }

    
    /// Called by MMU when it received data that this CPU is waiting for
    void CommitLoad(int data)
    {
        assert(IsStalling());

        // TODO: Separate LOAD instruction into two parts:
        //      1. The instruction handler calls MMU.LoadWord(address)
        //      2. This function is called by MMU upon request completion (might be immediate or might take a while)
        //          -> Execute the rest of the load instruction here 
        //          -> Note: This might run on a different thread, but it's OK, because StepOne cannot run while still stalling
        
        
        isStalling = false;
    }

    // ...
};


/// ARMulator calls Core "CPU", so we rename it accordingly
typedef CPU Core;