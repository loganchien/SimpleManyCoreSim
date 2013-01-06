#include "ArmulatorMMU.h"

#include "error.h"
#include "../../../src/MMU.hpp"

MMU::MMU(const char *file_name)
    : mmu(0)
{
}

MMU::~MMU()
{
}

//! Give out the Thumb instruction
T_INSTR MMU::getInstr(int address)
{
    T_INSTR instr = 0u;
    if (!mmu->LoadReadyHalfWord(address, instr))
    {
        throw LoadStall();
    }
    return instr;
}

//! Give out the ARM instruction
A_INSTR MMU::getInstr32(int address)
{
    A_INSTR instr = 0u;
    if (!mmu->LoadReadyWord(address, instr))
    {
        throw LoadStall();
    }
    return instr;
}

//! Output byte data by address
BYTE MMU::get_byte(int address)
{
    BYTE result = 0;
    if (!mmu->LoadReadyByte(address, result))
    {
        throw LoadStall();
    }
    return result;
}

//! Output halfword data by address
HALFWORD MMU::get_halfword(int address)
{
    HALFWORD result = 0;
    if (!mmu->LoadReadyHalfWord(address, result))
    {
        throw LoadStall();
    }
    return result;
}

//! Output word data by address
WORD MMU::get_word(int address)
{
    WORD result = 0;
    if (!mmu->LoadReadyWord(address, result))
    {
        throw LoadStall();
    }
    return result;
}

//! Input byte data by address
void MMU::set_byte(int address, BYTE data)
{
    mmu->StoreByte(address, data);
}

//! Input halfword data by address
void MMU::set_halfword(int address, HALFWORD data)
{
    mmu->StoreHalfWord(address, data);
}

//! Input word data by address
void MMU::set_word(int address, WORD data)
{
    mmu->StoreWord(address, data);
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
