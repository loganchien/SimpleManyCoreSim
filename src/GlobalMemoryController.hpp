#ifndef GLOBAL_MEMORY_CONTROLLER_HPP
#define GLOBAL_MEMORY_CONTROLLER_HPP

#include "Message.hpp"
#include "SimConfig.hpp"

#include <deque>
#include <stdint.h>

class elf_file;

namespace smcsim {

class Processor;

/// The global memory controller simulates transport of packets to and from
/// main memory
class GlobalMemoryController
{
public:
    /// The index of the gmem controller is unimportant, but can generally be
    /// used to identify gmem as a sender
    static const int GMemIdx = 1024 * 64 - 1;

    Processor* processor;

    uint8_t* text;
    uint32_t textVMA;
    uint32_t textSize;

    uint8_t* data;
    uint32_t dataVMA;
    uint32_t dataSize;

    uint8_t* rodata;
    uint32_t rodataVMA;
    uint32_t rodataSize;

    uint8_t* bss;
    uint32_t bssVMA;
    uint32_t bssSize;

    uint8_t* heap;
    uint32_t heapVMA;
    uint32_t heapSize;

    uint8_t* stackBegin;
    uint32_t stackBeginVMA;
    uint32_t stackBeginSize;

    uint8_t* stack;
    uint32_t stackVMA;

    uint32_t entryPointVMA;

    std::deque<Message> msgQueue;

    /// The sim time of the mem controller
    long long simTime;

public:
    GlobalMemoryController();

    void InitGMemController(Processor* processor);

    void Reset();

    void EnqueueRequest(const Message& msg);

    /// Load/Store Memory Image

    void LoadExecutable(std::ifstream &stream);

    void StoreCoreDump();

    /// Process the next gmem request
    void DispatchNext();

    /// Interface to access the memory
    uint8_t LoadByte(uint32_t addr);

    uint16_t LoadHalfWord(uint32_t addr);

    uint32_t LoadWord(uint32_t addr);

    void StoreByte(uint32_t addr, uint8_t byte);

    void StoreHalfWord(uint32_t addr, uint16_t halfword);

    void StoreWord(uint32_t addr, uint32_t word);

private:
    uint8_t* GetMemory(uint32_t addr, uint32_t size);
};

} // end namespace smcsim

#endif // GLOBAL_MEMORY_CONTROLLER_HPP
