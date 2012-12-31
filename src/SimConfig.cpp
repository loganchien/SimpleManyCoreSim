#include "SimConfig.hpp"

#include "Dimension.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

using namespace boost::property_tree;
using namespace boost::property_tree::ini_parser;
using namespace std;

SimConfig GlobalConfig;

SimConfig::SimConfig(): CoreGridLen(0), CoreBlockLen(0), CacheL1Size(0),
    CacheL1Delay(0), CacheL2Size(0), CacheL2Delay(0), CacheMissDelay(0),
    DispatchDelay(0), Route1Delay(0), MemDelay(0)
{
}

bool SimConfig::LoadConfig(const string& path)
{
    ifstream stream;
    stream.open(path.c_str(), ios_base::in);
    if (!stream)
    {
        return false;
    }
    return LoadConfig(stream);
}

bool SimConfig::LoadConfig(istream& stream)
{
    ptree pt;
    read_ini(stream, pt);

    CoreGridLen = pt.get("SETTING.CORE_GRID_LEN", 2);
    CoreBlockLen = pt.get("SETTING.CORE_BLOCK_LEN", 2);
    CacheL1Size = pt.get("SETTING.CACHE_L1_SIZE", 32 * 1024);
    CacheL1Delay = pt.get("SETTING.CAChe_L2_DELAY", 1);
    CacheL2Size = pt.get("SETTING.CACHE_L2_SIZE", 256 * 1024);
    CacheL2Delay = pt.get("SETTING.CACHE_L2_DELAY", 5);
    CacheMissDelay = pt.get("SETTING.CACHE_MISS_DELAY", 50);
    DispatchDelay = pt.get("SETTING.DISPATCH_DELAY", 5);
    Route1Delay = pt.get("SETTING.ROUTE_1_DELAY", 10);
    MemDelay = pt.get("SETTING.MEM_DELAY", 50);

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

int SimConfig::ComputeInCoreBlockIdx1(const Dim2& inCoreBlockIdx2)
{
	return inCoreBlockIdx2.y * CoreBlockLen + inCoreBlockIdx2.x;
}

int SimConfig::GetTotalL2CacheSize()
{
    return CoreBlockSize().Area() * CacheL2Size;
}
