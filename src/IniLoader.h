#include "SimpleIni.h"
#include "SimConfig.hpp"
#include <vector>

using namespace std;

class IniLoader {
public:
	vector<SimConfig> loadConfigs();

};