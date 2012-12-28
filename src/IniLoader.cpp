#include "IniLoader.h"

// TODO: Load several different configs from file
vector<SimConfig> loadConfigs(){
	CSimpleIniA ini;
	ini.SetUnicode();
	ini.LoadFile(".\\ini\\example.ini");

	vector<SimConfig> configs;
	SimConfig sc;
	configs[0] = sc;
	sc.CacheL1Delay = (int)ini.GetLongValue("Setting", "CACHE_L1_DELAY", NULL);
	sc.CoreGridLen = (int)ini.GetLongValue("Setting", "CORE_GRID_LEN", NULL);
	//...
	//example if we need string read out from ini:
	//const char * pVal = ini.GetValue("Setting", "CORE_GRID_LEN", NULL);

	return configs;
}