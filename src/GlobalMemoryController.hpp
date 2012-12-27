#ifndef GLOBAL_MEMORY_CONTROLLER_HPP
#define GLOBAL_MEMORY_CONTROLLER_HPP

#include "Message.hpp"
#include "SimConfig.hpp"

#include <deque>
#include <stdint.h>

class Processor;

/// The global memory controller simulates transport of packets to and from main memory
class GlobalMemoryController
{
public:
    /// The index of the gmem controller is unimportant, but can generally be used to identify gmem as a sender
    static const int GMemIdx = 1024 * 64 - 1;

    Processor* processor;

    std::vector<uint32_t> memory;
    std::deque<Message> msgQueue;

    /// The sim time of the mem controller
    long long simTime;

    GlobalMemoryController();

    void InitGMemController(Processor* processor);

    void EnqueueRequest(const Message& msg);

    /// Process the next gmem request
    void DispatchNext();
};

#endif // GLOBAL_MEMORY_CONTROLLER_HPP
