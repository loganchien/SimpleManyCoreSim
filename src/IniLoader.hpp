#ifndef INILOADER_H
#define INILOADER_H

#include "SimpleIni.h"
#include "SimConfig.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

class IniLoader {
public:
	static vector<SimConfig> loadConfigs(){
		CSimpleIniA ini;
		vector<SimConfig> configs;
		SimConfig* sc;
		ini.SetUnicode();

		//TODO: automatic loading of all ini files
		// use txt file with all files is the easiest method, 
		// (navigate to ini folder first!) add to the MAKE cmd: dir /b > list.txt for windows, ls > list.txt for linux?

		string s,file;
		int i = 0;
		char* fileName;
		ifstream infile;
		infile.open("list.txt");
		while(!infile.eof()) // To get all files
		{
			file=".\\ini\\";
			getline(infile,s);
			if (s.find(".ini") != string::npos) { //check if it's an .ini file
				file.append(s);
				fileName = (char*)file.c_str();		
				ini.LoadFile(fileName);
				configs.reserve(i+1);
				configs.insert(configs.begin()+i,SimConfig());
				configs[i].CoreGridLen = (int)ini.GetLongValue("Setting", "CORE_GRID_LEN", NULL);
				configs[i].CoreBlockLen = (int)ini.GetLongValue("Setting", "CORE_BLOCK_LEN", NULL);
				configs[i].CacheL1Size = (int)ini.GetLongValue("Setting", "CACHE_L1_SIZE", NULL);
				configs[i].CacheL1Delay = (int)ini.GetLongValue("Setting", "CACHE_L1_DELAY", NULL);
				configs[i].CacheL2Size = (int)ini.GetLongValue("Setting", "CACHE_L2_SIZE", NULL);
				configs[i].CacheL1Delay = (int)ini.GetLongValue("Setting", "CACHE_L2_DELAY", NULL);
				configs[i].CacheMissDelay = (int)ini.GetLongValue("Setting", "CACHE_MISS_DELAY", NULL);
				configs[i].DispatchDelay = (int)ini.GetLongValue("Setting", "DISPATCH_DELAY", NULL);
				configs[i].Route1Delay = (int)ini.GetLongValue("Setting", "ROUTE_1_DELAY", NULL);
				configs[i].MemDelay = (int)ini.GetLongValue("Setting", "MEM_DELAY", NULL);
				i++;
			}
		}
		infile.close();

		//example if we need string read out from ini:
		//const char * pVal = ini.GetValue("Setting", "CORE_GRID_LEN", NULL);

		return configs;
	}
};

#endif