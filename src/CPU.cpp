#include "CPU.hpp"

#include "Address.hpp"

#include <assert.h>
#include <stdint.h>

using namespace smcsim;

Core::Core(Tile* tile)
{
    this->tile = tile;

    // TODO: mmu.InitMMU(thread->tile);
}


/// This Core starts running the given Thread
void Core::StartThread(Thread* thread)
{
    currentThread = thread;

    isLoadingData = false;
    simInstructionCount = simLoadInstructionCount = 0;

    mmu.ResetMMU();

    // TODO: Copy thread.code into mmu.code
}


/// Dispatches and, if possible, executes one simulated instruction. Returns false, if there are no more instructions to execute (i.e. EOF reached).
bool Core::DispatchNext()
{
    if (isLoadingData) return true;

    ++simInstructionCount;

    // TODO: Only run/start a single instruction

    // TODO: Call DispatchLoad when encountering a load instruction

    // TODO: If the current instruction was the last instruction (or EOF), call tile->coreBlock->OnThreadFinished(*currentThread);
}


/// Forwards a load instruction to the MMU
void Core::DispatchLoad(int addrWord)
{
    ++simLoadInstructionCount;
    Address addr(addrWord);
    mmu.LoadWord(addr);
    isLoadingData = true;
}


/// Called by MMU when it received data that this Core is waiting for
void Core::CommitLoad(uint32_t data)
{
    assert(isLoadingData);

    // TODO: Separate LOAD instruction into two parts:
    //      1. The instruction handler calls MMU.LoadWord(address)
    //      2. This function is called by MMU upon request completion (might be immediate or might take a while)
    //          -> Execute the rest of the load instruction here

    // TODO: Figure out which part of the requested word is needed (which byte, which half-word, or the entire word?)
    //      Possibly by just storing the requested length in a variable before in DispatchLoad

    isLoadingData = false;
}
