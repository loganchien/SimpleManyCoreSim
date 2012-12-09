
#include "Message.hpp"

#include <mutex>
#include <deque>

#include "SimConfig.hpp"


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

    void DispatchNext()
    {
        if (msgQueue.size() == 0) return;
        
        // Dequeue next message
        queueLock.lock();
        Message& request = msgQueue.front();
        msgQueue.pop_front();
        queueLock.unlock();

        // send response back to sender
        Message response;
        response.type = MessageTypeResponseCacheline;
        response.sender = int2(GMemIdx, GMemIdx);               // this does not matter
        response.receiver = request.sender;                     
        response.requestId = request.requestId;
        response.totalDelay = request.totalDelay + GlobalConfig.MemDelay;
        memcpy(response.cacheLine, &memory[request.addr.Raw], sizeof(WORD) * GlobalConfig::CacheLineSize);

        // TODO: Compute index of boundary router, closest to destination router
        int2 nearestRouterId = ;
        
        Router& nearestRouter = processor.GetTile(nearestRouterId).router;

        // Send out new response Message
        nearestRouter.EnqueueMessage(response);
    }
};