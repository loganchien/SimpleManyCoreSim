#ifndef SIMUTIL_HPP
#define SIMUTIL_HPP

typedef unsigned short ushort;

#include <cassert>
#include <cmath>
#include <iostream>

// include config, make it available to everyone
//#include "SimConfig.hpp"

// Some convenient macros
#define PrintLine(str) std::cout << str << std::endl;

/// A pair of (small) integers, x and y
struct int2
{
    ushort x, y;

    int2() : x(0), y(0) {}
    int2(int x, int y) : x(x), y(y) {}

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

    bool operator==(const int2 &rhs) const {
        return (x == rhs.x && y == rhs.y);
    }
};

/// Represents an address in 32-bit address space
struct Address
{
    /// The raw integer representation of the address
    unsigned int addr;

    union
    {
        unsigned int L1Index;
        unsigned int L1Tag;
    };

    union
    {
        unsigned int L2Index;
        unsigned int L2Tag;
    };

    /// The offset of the word that this address is referring to
    int WordOffset;

    Address(int addr = 0)
    {
        // TODO: Read index, tag and offset from address
    }

    /// The chunk of the shared L2 cache, to which this address maps
    int GetL2ChunkIndex() const; // FIXME: Check return type?
};

#endif // SIMUTIL_HPP
