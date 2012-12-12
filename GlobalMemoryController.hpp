#ifndef GLOBAL_MEMORY_CONTROLLER_HPP
#define GLOBAL_MEMORY_CONTROLLER_HPP

#include "SimConfig.hpp"
#include "Message.hpp"

#include <mutex>
#include <deque>
#include <vector>

#include <string.h>

struct Processor;
struct Router;

/// The global memory controller simulates transport of packets to and from main memory
struct GlobalMemoryController
{
    /// The index of the gmem controller is unimportant, but can generally be used to identify gmem as a sender
    static const int GMemIdx = 1024 * 64 - 1;

    Processor* processor;

    std::vector<WORD> memory;
    std::mutex queueLock;
    std::deque<Message> msgQueue;

    /// The sim time of the mem controller
    long long simTime;

    GlobalMemoryController()
    {
        memory.resize(MAX_MEM_SIZE);
    }

    void InitGMemController(Processor* processor)
    {
        this->processor = processor;
    }

    void EnqueueRequest(const Message& msg)
    {
        // Add to queue
        queueLock.lock();
        msgQueue.push_back(msg);
        queueLock.unlock();
    }

    void DispatchNext();
};

#endif // GLOBAL_MEMORY_CONTROLLER_HPP
