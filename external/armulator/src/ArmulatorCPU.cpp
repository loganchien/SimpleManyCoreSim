#include "ArmulatorCPU.h"

#include "ARM.h"
#include "CPU.h"
#include "Thumb.h"
#include "error.h"

#include <iostream>

ArmulatorCPU::ArmulatorCPU(): cpu(0)
{
}

ArmulatorCPU::~ArmulatorCPU()
{
    reset();
}

void ArmulatorCPU::init()
{
    cpu = new ARM();
}

void ArmulatorCPU::reset()
{
    if (cpu)
    {
        cpu->DeinitMMU();
        delete cpu;
        cpu = 0;
    }
}

void ArmulatorCPU::sim_step()
{
    try
    {
        cpu->fetch();
        cpu->exec();
    }
    catch (Error &e)
    {
        cpu->DeinitMMU();
        std::cerr << "\nError:" << e.error_name <<std::endl;
    }
    catch (UnexpectInst &e)
    {
        std::cerr << "\nUnexpect Instr:" << e.error_name << std::endl;
    }
    catch (UndefineInst &e)
    {
        std::cerr << "\nUndefine Instr:" << e.error_name << std::endl;
    }
    catch (SwitchMode &e)
    {
        Thumb *tmp = new Thumb();
        tmp->CopyCPU(cpu);
        delete cpu;
        cpu = tmp;
    }
    catch (ProgramEnd &e)
    {
        std::cerr << "\nThe Program Ended\n";
    }
}
