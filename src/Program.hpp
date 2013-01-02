#ifndef PROGRAM_HPP
#define PROGRAM_HPP

#include "Address.hpp"
#include <string>

namespace smcsim {

class Program
{
public:
    std::string elfFilePath;
    Address threadIdxAddress;
    Address threadDimAddress;
    Address blockIdxAddress;
    Address blockDimAddress;

public:
    Program();
};

} // end namespace smcsim

#endif // PROGRAM_HPP
