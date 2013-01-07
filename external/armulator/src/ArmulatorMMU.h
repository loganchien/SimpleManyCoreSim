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

    MMU(smcsim::MMU *tileMMU);

    ~MMU();

    //! Give out the Thumb instruction
    T_INSTR getInstr(int address, bool simulate_delay = false);

    //! Give out the ARM instruction
    A_INSTR getInstr32(int address, bool simulate_delay = false);

    //! Output byte data by address
    BYTE get_byte(int address, bool simulate_delay = false);

    //! Output halfword data by address
    HALFWORD get_halfword(int address, bool simulate_delay = false);

    //! Output word data by address
    WORD get_word(int address, bool simulate_delay = false);

    //! Input byte data by address
    void set_byte(int address, BYTE data, bool simulate_delay = false);

    //! Input halfword data by address
    void set_halfword(int address, HALFWORD data, bool simulate_delay = false);

    //! Input word data by address
    void set_word(int address, WORD data, bool simulate_delay = false);

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
