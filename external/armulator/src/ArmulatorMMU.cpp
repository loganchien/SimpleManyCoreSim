#include "ArmulatorMMU.h"

MMU::MMU(const char *file_name)
{
}

MMU::~MMU()
{
}

//! Give out the Thumb instruction
T_INSTR MMU::getInstr(int address)
{
    return 0;
}

//! Give out the ARM instruction
T_INSTR MMU::getInstr32(int address)
{
    return 0;
}

//! Output byte data by address
BYTE MMU::get_byte(int address)
{
    return 0;
}

//! Output halfword data by address
HALFWORD MMU::get_halfword(int address)
{
    return 0;
}

//! Output word data by address
WORD MMU::get_word(int address)
{
    return 0;
}

//! Input byte data by address
void MMU::set_byte(int address, BYTE data)
{
}

//! Input halfword data by address
void MMU::set_halfword(int address, HALFWORD data)
{
}

//! Input word data by address
void MMU::set_word(int address, WORD data)
{
}


//! Give out the starting PC
int MMU::getEntry()
{
    return 0;
}

//! Give out the high address of stack
int MMU::getStackTop()
{
    return 0;
}

//! Give out the size of stack
int MMU::getStackSz()
{
    return 0;
}

//! Give out the address of heap
int MMU::getHeapTop()
{
    return 0;
}

//! Give out the size of heap
int MMU::getHeapSz()
{
    return 0;
}
