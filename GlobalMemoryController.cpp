#include "GlobalMemoryController.hpp"

/// FIXME: Enable this after Processor.hpp is fixed.
#if 0
#include "Processor.hpp"
#include "Router.hpp"

void GlobalMemoryController::DispatchNext()
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
    memcpy(response.cacheLine, &memory[request.addr.addr], sizeof(WORD) * SimConfig::CacheLineWordSize);

    // TODO: Compute index of boundary router, closest to destination router
    int2 nearestRouterId; // FIXME: Should not be zero

    Router& nearestRouter = processor->GetTile(nearestRouterId).router;

    // Send out new response Message
    nearestRouter.EnqueueMessage(response);
}
#endif
