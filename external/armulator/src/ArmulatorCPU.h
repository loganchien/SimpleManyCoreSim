#ifndef __ARMULATOR_CPU__
#define __ARMULATOR_CPU__

namespace smcsim {
    class Core;
}

class CPU;

class ArmulatorCPU
{
private:
    //! The armulator cpu (either ARM or Thumb)
    CPU *cpu;

    //! The SimpleManyCoreSim core to which this CPU belongs
    smcsim::Core *core;

public:
    ArmulatorCPU();

    ~ArmulatorCPU();

    void init();

    void reset();

    bool sim_step();
};

#endif // __ARMULATOR_CPU__
