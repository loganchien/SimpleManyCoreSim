#ifndef __TILE_MMU__
#define __TILE_MMU__

#include "arch.h"

namespace smcsim {
    class MMU;
} // end namespace smcsim

class MMU
{
private:
    smcsim::MMU *mmu;

public:
    MMU(const char *file_name);

    ~MMU();

    //! Give out the Thumb instruction
    T_INSTR getInstr(int address);

    //! Give out the ARM instruction
    A_INSTR getInstr32(int address);

    //! Output byte data by address
    BYTE get_byte(int address);

    //! Output halfword data by address
    HALFWORD get_halfword(int address);

    //! Output word data by address
    WORD get_word(int address);

    //! Input byte data by address
    void set_byte(int address, BYTE data);

    //! Input halfword data by address
    void set_halfword(int address, HALFWORD data);

    //! Input word data by address
    void set_word(int address, WORD data);

    //! Give out the starting PC
    int getEntry();

    //! Give out the high address of stack
    int getStackTop();

    //! Give out the size of stack
    int getStackSz();

    //! Give out the address of heap
    int getHeapTop();

    //! Give out the size of heap
    int getHeapSz();
};

#endif // __TILE_MMU__
