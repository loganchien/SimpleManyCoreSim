#include "ArmulatorCPU.h"

#include "ARM.h"
#include "ArmulatorMMU.h"
#include "CPU.h"
#include "Thumb.h"
#include "error.h"

#include "../../../src/Core.hpp"

#include <iostream>

#include <stdlib.h>

ArmulatorCPU::ArmulatorCPU(): cpu(0)
{
}

ArmulatorCPU::~ArmulatorCPU()
{
    reset();
}

void ArmulatorCPU::init(smcsim::MMU *tileMMU)
{
    cpu = new ARM();
    cpu->InitMMU(new MMU(tileMMU));
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

bool ArmulatorCPU::sim_step()
{
    assert(cpu);
    try
    {
        cpu->fetch();
        cpu->exec();
        return true;
    }
    catch (Error &e)
    {
        std::cerr << "\nError:" << e.error_name <<std::endl;
        abort();
    }
    catch (UnexpectInst &e)
    {
        std::cerr << "\nUnexpect Instr:" << e.error_name << std::endl;
        abort();
    }
    catch (UndefineInst &e)
    {
        std::cerr << "\nUndefine Instr:" << e.error_name << std::endl;
        abort();
    }
    catch (LoadStall &e)
    {
        cpu->unfetch();
        return true;
    }
    catch (SwitchMode &e)
    {
        Thumb *tmp = new Thumb();
        tmp->CopyCPU(cpu);
        delete cpu;
        cpu = tmp;
        return true;
    }
    catch (ProgramEnd &e)
    {
        reset();
        return false;
    }

    return true;
}
