#include "Core.hpp"

#include "Address.hpp"
#include "CoreBlock.hpp"
#include "Debug.hpp"
#include "Tile.hpp"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

using namespace smcsim;

Core::Core(Tile* tile_)
    : tile(tile_), currentThread(NULL), loadStallDelay(0),
      simInstructionCount(0), simLoadInstructionCount(0)
{
}


/// This Core starts running the given Thread
void Core::StartThread(Thread* thread)
{
    // Set the running thread
    currentThread = thread;

    // Reset the simulation counter
    simInstructionCount = 0;
    simLoadInstructionCount = 0;

    // Initialize the core
    armulator.init(this, &tile->mmu);
}


/// Dispatches and, if possible, executes one simulated instruction. Returns
/// false, if there are no more instructions to execute (i.e. EOF reached).
bool Core::DispatchNext()
{
    if (!currentThread)
    {
        return false;
    }
    if (loadStallDelay > 0)
    {
        --loadStallDelay;
        return true;
    }
    ++simInstructionCount;

    if (!armulator.sim_step())
    {
        // Program ends.
        assert(tile);
        assert(tile->coreBlock);
        assert(currentThread);

        Thread* finishedThread = currentThread;
        currentThread = NULL;
        tile->coreBlock->OnThreadFinished(*finishedThread);

        return false;
    }

    return true;
}


void Core::OnLoadStall(int delay)
{
    assert(delay >= 0);
    loadStallDelay = delay;
}


/// Called by MMU when it received data that this Core is waiting for
void Core::CommitLoad(uint32_t data)
{
    // TODO: Separate LOAD instruction into two parts:
    // 1. The instruction handler calls MMU.LoadWord(address)
    // 2. This function is called by MMU upon request completion (might be
    //    immediate or might take a while) -> Execute the rest of the load
    //    instruction here

    // TODO: Figure out which part of the requested word is needed (which byte,
    // which half-word, or the entire word?) Possibly by just storing the
    // requested length in a variable before in DispatchLoad
}
