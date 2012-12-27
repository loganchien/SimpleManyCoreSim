#ifndef PROGRAM_HPP
#define PROGRAM_HPP

#include "Address.hpp"
#include <string>

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

#endif // PROGRAM_HPP
