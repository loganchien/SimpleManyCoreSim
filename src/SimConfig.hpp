#ifndef SIM_CONFIG_HPP
#define SIM_CONFIG_HPP

#include "Dimension.hpp"

#include <iostream>
#include <string>

namespace smcsim {

/**
 * All tweakable parameters of the system
 */

#define MAX_MEM_SIZE (8 * 1024 * 1024)

class SimConfig
{
public:
    // #################### Memory ############################################

    /// Stack size per core (bytes)
    int StackSize;

    // #################### Tile Grid #########################################

    /// Length of the grid of core blocks
    int CoreGridLen;

    /// Total amount of core blocks on the processor
    Dim2 CoreGridSize();

    /// Total amount of cores (or tiles) in the processor
    int TotalCoreLength();

    // #################### Core Blocks #######################################

    /// Side length of a core block. If CoreBlockLen == 1, then we have private
    /// L2s!
    int CoreBlockLen;

    /// Total amount of cores in a core block
    Dim2 CoreBlockSize();

    /// Converts the given block-local 1D index to the corresponding
    /// block-local 2D index
    Dim2 ComputeInCoreBlockIdx2(int inCoreBlockIdx1);

    /// Converts the given block-local 2D index to the corresponding
    /// block-local 1D index
    int ComputeInCoreBlockIdx1(const Dim2& inCoreBlockIdx2);


    // #################### Caches ############################################

    /// 64 bytes per cache line (64 bytes = 16 x 4 byte words)
    static const int CacheLineSize = 64;
    static const int numbCacheLineBits = 6; //log2(64)

    static const int CacheLineBits = 0x3f;

    /// L1 size & access time
    int CacheL1Size, CacheL1Delay;

    /// L2 size & access time
    int CacheL2Size, CacheL2Delay;

    /// Main memory access time
    int CacheMissDelay;

    /// Total size of shared L2 in a core block
    int GetTotalL2CacheSize();

    // #################### Networking ########################################

    /// Delay when a router processes a packet
    int DispatchDelay;

    /// Delay when sending a packet from one Router to a neighbor Router
    int Route1Delay;

    ///
    int MemDelay;

    // Simulate queueing delay for all waiting packets
    int QueuingDelay;

public:
    SimConfig();

    bool LoadConfig(const std::string& path);
};

extern SimConfig GlobalConfig;

} // end namespace smcsim

#endif // SIM_CONFIG_HPP
