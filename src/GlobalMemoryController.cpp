#include "GlobalMemoryController.hpp"

#include "Address.hpp"
#include "Processor.hpp"
#include "Router.hpp"
#include "SimConfig.hpp"

#include <string.h>

using namespace smcsim;

GlobalMemoryController::GlobalMemoryController()
{
    memory.resize(MAX_MEM_SIZE);
}

void GlobalMemoryController::InitGMemController(Processor* processor_)
{
    processor = processor_;
}

void GlobalMemoryController::EnqueueRequest(const Message& msg)
{
    // Add to queue
    msgQueue.push_back(msg);
}

void GlobalMemoryController::DispatchNext()
{
    if (msgQueue.size() == 0) return;

    // Dequeue next message
    Message& request = msgQueue.front();
    msgQueue.pop_front();

    // send response back to sender
    Message response;
    response.type = MessageTypeResponseCacheline;
    response.sender = Dim2(GMemIdx, GMemIdx);               // this does not matter
    response.receiver = request.sender;
    response.requestId = request.requestId;
    response.totalDelay = request.totalDelay + GlobalConfig.MemDelay;
    memcpy(&*response.cacheLine.bytes.begin(), &memory[request.addr.raw], sizeof(uint32_t) * GlobalConfig.CacheLineSize);

    // TODO: Compute index of boundary router, closest to destination router
    Dim2 nearestRouterId;

    Router nearestRouter; // TODO: processor->getTile(nearestRouterId).router;

    // Send out new response Message
    nearestRouter.EnqueueMessage(response);
}
