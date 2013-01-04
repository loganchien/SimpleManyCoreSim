#ifndef __ARMULATOR_CPU__
#define __ARMULATOR_CPU__

class CPU;

class ArmulatorCPU
{
private:
    CPU *cpu;

public:
    ArmulatorCPU();

    ~ArmulatorCPU();

    void init();

    void reset();

    void sim_step();
};

#endif // __ARMULATOR_CPU__
