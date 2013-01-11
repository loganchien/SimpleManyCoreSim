#ifndef ADDRESS_HPP
#define ADDRESS_HPP

#include <stdint.h>

namespace smcsim {

/// Represents an address in 32-bit address space
class Address
{
public:
    /// The raw integer representation of the address
    uint32_t raw;

public:
    Address(uint32_t addr = 0u): raw(addr)
    { }
	uint32_t GetIndex(int mask) const;
    uint32_t GetTag(int shift) const;

    uint32_t GetL1Index() const;
    uint32_t GetL1Tag() const;

    uint32_t GetL2Index() const;
    uint32_t GetL2Tag() const;

    /// The offset of the word that this address is referring to
    uint32_t GetWordOffset() const;

    /// The block-local L2 chunk index, to which this address maps
    int GetL2ChunkIdx1() const;
};

} // end namespace smcsim

#endif // ADDRESS_HPP
