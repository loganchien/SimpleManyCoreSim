#ifndef SIM_CONFIG_HPP
#define SIM_CONFIG_HPP

#include "VisualCXXCompatibility.hpp"
#include "simutil.hpp"

#include <stdlib.h>

/**
 * All tweakable parameters of the system
 */

#define _VERBOSE

#define MAX_MEM_SIZE (8 * 1024 * 1024)

struct SimConfig
{
    // ################################ Tile Grid ################################

    /// Length of the grid of core blocks
    int CoreGridLen;

    /// Total amount of core blocks on the processor
    int CoreGridSize() { return CoreGridLen * CoreGridLen; }



    // ################################ Core Blocks ################################

    /// Side length of a core block. If CoreBlockLen == 1, then we have private L2s!
    int CoreBlockLen;

    /// Total amount of cores in a core block
    int CoreBlockSize() { return CoreBlockLen * CoreBlockLen; }

    /// Convers the given 1D index within a CoreBlock to the corresponding 2D index
    int2 ComputeInCoreBlockIdx2(int inCoreBlockIdx1)
    {
        div_t d = div(inCoreBlockIdx1, CoreBlockLen);
        return int2(d.quot, d.rem);
    }


    // ################################ Caches ################################

    /// 64 bytes per cache line (= 16 x 4 byte words)
    static const int CacheLineWordSize = 16;

    /// L1 size & access time
    int CacheL1Size, CacheL1Delay;

    /// L2 size & access time
    int CacheL2Size, CacheL2Delay;

    /// Main memory access time
    int CacheMissDelay;

    /// Total size of shared L2 in a core block
    int GetTotalL2CacheSize()
    {
        return CoreBlockSize() * CacheL2Size;
    }

    // ################################ Networking ################################

    /// Delay when a router processes a packet
    int DispatchDelay;

    /// Delay when sending a packet from one Router to a neighbor Router
    int Route1Delay;

    ///
    int MemDelay;
};

extern SimConfig GlobalConfig;

#endif // SIM_CONFIG_HPP
