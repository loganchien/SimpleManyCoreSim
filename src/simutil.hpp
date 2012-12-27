/**
 * This file is included by all classes
 */

// include guard
#ifndef _SMUTIL_HPP
#define _SMUTIL_HPP

typedef unsigned short ushort;

// assert
#include <cassert>
#include <cmath>
#include <iostream>


// Some convenient macros
#define PrintLine(str) std::cout << str << endl;

typedef unsigned int uint;


/// A pair of (small) integers, x and y
typedef struct int2
{
    ushort x, y;

    int2(int x = 0, int y = 0) : x(x), y(y) {}

    int Area() const { return (int)x * (int)y; }
     
    /// Convert to one-dimensional index
    int Get1DIndex(int width) const { return y * width + x; }

    /// Increments x by one, but wraps x at width, at which point, it resets x and increments y by one.
    void Inc(int width)
    {
        ++x;

        if (x == width)
        {
            x = 0;
            ++y;
        }
    }


    /// Component-wise addition
    int2 operator+ (int2 rhs)
    {
        return int2(x + rhs.x, y + rhs.y);
    }


    /// Component-wise subtraction
    int2 operator- (int2 rhs)
    {
        return int2(x - rhs.x, y - rhs.y);
    }
    

    /// Component-wise multiplication
    int2 operator* (int2 rhs)
    {
        return int2(x * rhs.x, y * rhs.y);
    }
} int2;


// include config, make it available to everyone
#include "SimConfig.hpp"

/// Represents an address in 32-bit address space
struct Address
{
    /// The raw integer representation of the address
    uint Raw;


    uint GetL1Index() const { return GlobalConfig.CacheL1Size TODO; }
    uint GetL1Tag() const { return TODO; }
    
    uint GetL2Index() const { return TODO; }
    uint GetL2Tag() const { return TODO; }
    
    /// The offset of the word that this address is referring to
    uint GetWordOffset() const
    {
        return Raw & GlobalConfig.CacheLineBits;
    }
    
    Address(uint raw = 0)
    {
        // TODO: Read index, tag and offset from address
    }


    /// The block-local L2 chunk index, to which this address maps
    int GetL2ChunkIdx1() const
    {
        return (GetL2Index() / GlobalConfig.CacheL2Size) % GlobalConfig.CoreBlockSize();
    }
};

#endif