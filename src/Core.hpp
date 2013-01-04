#ifndef CPU_HPP
#define CPU_HPP

#include "Dimension.hpp"
#include "MMU.hpp"

#include <stdint.h>

namespace smcsim {

class Tile;
class Thread;

class Core
{
public:
    /// The tile to which this Core belongs
    Tile* tile;

    /// The currently running thread
    Thread* currentThread;

    /// Whether this Core is currently waiting for off-chip data
    bool isLoadingData;

    /// The total amount of instructions
    long long simInstructionCount, simLoadInstructionCount;

    // ...
    MMU mmu;


public:
    Core(Tile* tile);


    /// This Core starts running the given Thread
    void StartThread(Thread* thread);


    /// Dispatches and, if possible, executes one simulated instruction. Returns false, if there are no more instructions to execute (i.e. EOF reached).
    bool DispatchNext();


    /// Forwards a load instruction to the MMU
    void DispatchLoad(int addrWord);


    /// Called by MMU when it received data that this Core is waiting for
    void CommitLoad(uint32_t data);
};

} // end namespace smcsim

#endif // CPU_HPP
