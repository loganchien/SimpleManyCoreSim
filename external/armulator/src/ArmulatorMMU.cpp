#include "ArmulatorMMU.h"

#include "error.h"
#include "../../../src/MMU.hpp"

MMU::MMU(const char *file_name)
    : mmu(0)
{
}

MMU::MMU(smcsim::MMU* tileMMU)
    : mmu(tileMMU)
{
}

MMU::~MMU()
{
}

//! Give out the Thumb instruction
T_INSTR MMU::getInstr(int address, bool simulate_delay)
{
    return mmu->GetHalfWord(address, false);
}

//! Give out the ARM instruction
A_INSTR MMU::getInstr32(int address, bool simulate_delay)
{
    return mmu->GetWord(address, false);
}

//! Output byte data by address
BYTE MMU::get_byte(int address, bool simulate_delay)
{
    return mmu->GetByte(address, simulate_delay);
}

//! Output halfword data by address
HALFWORD MMU::get_halfword(int address, bool simulate_delay)
{
    return mmu->GetHalfWord(address, simulate_delay);
}

//! Output word data by address
WORD MMU::get_word(int address, bool simulate_delay)
{
    return mmu->GetWord(address, simulate_delay);
}

//! Input byte data by address
void MMU::set_byte(int address, BYTE data, bool simulate_delay)
{
    mmu->SetByte(address, data, simulate_delay);
}

//! Input halfword data by address
void MMU::set_halfword(int address, HALFWORD data, bool simulate_delay)
{
    mmu->SetHalfWord(address, data, simulate_delay);
}

//! Input word data by address
void MMU::set_word(int address, WORD data, bool simulate_delay)
{
    mmu->SetWord(address, data, simulate_delay);
}


//! Give out the starting PC
int MMU::getEntry()
{
    return mmu->GetEntry();
}

//! Give out the high address of stack
int MMU::getStackTop()
{
    return mmu->GetStackTop();
}

//! Give out the size of stack
int MMU::getStackSz()
{
    return mmu->GetStackSize();
}

//! Give out the address of heap
int MMU::getHeapTop()
{
    return mmu->GetHeapTop();
}

//! Give out the size of heap
int MMU::getHeapSz()
{
    return mmu->GetHeapSize();
}
