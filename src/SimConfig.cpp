#include "SimConfig.hpp"

#include "Debug.hpp"
#include "Dimension.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

using namespace boost::property_tree;
using namespace boost::property_tree::ini_parser;
using namespace smcsim;
using namespace std;

SimConfig smcsim::GlobalConfig;

SimConfig::SimConfig()
    : StackSize(0), HeapSize(0),
      CoreGridLen(0), CoreBlockLen(0), CacheL1Size(0),
      CacheL1Delay(0), CacheL2Size(0), CacheL2Delay(0), CacheMissDelay(0),
      DispatchDelay(0), Route1Delay(0), MemDelay(0), QueuingDelay(0)
{
}

bool SimConfig::LoadConfig(const string& path)
{
    ptree pt;
    read_ini(path, pt);

    StackSize = pt.get("setting.STACK_SIZE", 1 * 1024 * 1024);
    HeapSize = pt.get("setting.HEAP_SIZE", 1 * 1024 * 1024);
    CoreGridLen = pt.get("setting.CORE_GRID_LEN", 2);
    CoreBlockLen = pt.get("setting.CORE_BLOCK_LEN", 2);
    CacheL1Size = pt.get("setting.CACHE_L1_SIZE", 32 * 1024);
    CacheL1Delay = pt.get("setting.CACHE_L1_DELAY", 1);
    CacheL2Size = pt.get("setting.CACHE_L2_SIZE", 256 * 1024);
    CacheL2Delay = pt.get("setting.CACHE_L2_DELAY", 5);
    CacheMissDelay = pt.get("setting.CACHE_MISS_DELAY", 50);
    DispatchDelay = pt.get("setting.DISPATCH_DELAY", 5);
    Route1Delay = pt.get("setting.ROUTE_1_DELAY", 10);
    MemDelay = pt.get("setting.MEM_DELAY", 50);
    QueuingDelay = pt.get("setting.QUEUING_DELAY", 1);

    PrintLine("CONFIG: Core Grid Len:    " << CoreGridLen);
    PrintLine("CONFIG: Core Block Len:   " << CoreBlockLen);
    PrintLine("CONFIG: L1 Cache Size:    " << CacheL1Size);
    PrintLine("CONFIG: L1 Cache Delay:   " << CacheL1Delay);
    PrintLine("CONFIG: L2 Cache Size:    " << CacheL2Size);
    PrintLine("CONFIG: L2 Cache Delay:   " << CacheL2Delay);
    PrintLine("CONFIG: Cache Miss Delay: " << CacheMissDelay);
    PrintLine("CONFIG: Dispatch Delay:   " << DispatchDelay);
    PrintLine("CONFIG: Route 1 Delay:    " << Route1Delay);
    PrintLine("CONFIG: Mem Delay:        " << MemDelay);
    PrintLine("CONFIG: Queuing Delay:    " << QueuingDelay);

    return true;
}

Dim2 SimConfig::CoreGridSize()
{
    return Dim2(CoreGridLen, CoreGridLen);
}

Dim2 SimConfig::CoreBlockSize()
{
    return Dim2(CoreBlockLen, CoreBlockLen);
}

Dim2 SimConfig::ComputeInCoreBlockIdx2(int inCoreBlockIdx1)
{
    div_t d = div(inCoreBlockIdx1, CoreBlockLen);
    return Dim2(d.quot, d.rem);
}

int SimConfig::TotalCoreLength(){
	return (CoreBlockLen*CoreGridLen);
}

int SimConfig::ComputeInCoreBlockIdx1(const Dim2& inCoreBlockIdx2)
{
	return inCoreBlockIdx2.y * CoreBlockLen + inCoreBlockIdx2.x;
}

int SimConfig::GetTotalL2CacheSize()
{
    return CoreBlockSize().Area() * CacheL2Size;
}
