#ifndef SIM_UTIL_HPP
#define SIM_UTIL_HPP

// assert
#include <cassert>
#include <cmath>
#include <iostream>

// Some convenient macros
#define PrintLine(str) std::cout << str << endl;

typedef unsigned short ushort;
typedef unsigned int uint;


// include config, make it available to everyone
#include "SimConfig.hpp"

/// Represents an address in 32-bit address space
class Address
{
public:
    /// The raw integer representation of the address
    uint Raw;

    uint GetL1Index() const;
    uint GetL1Tag() const;

    uint GetL2Index() const;
    uint GetL2Tag() const;

    /// The offset of the word that this address is referring to
    uint GetWordOffset() const;

    Address(uint raw = 0);

    /// The block-local L2 chunk index, to which this address maps
    int GetL2ChunkIdx1() const;
};

#endif // SIM_UTIL_HPP
