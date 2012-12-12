#include "Router.hpp"

// FIXME: Enable this after Processor.hpp is fixed.
#if 0
#include "Tile.hpp"
#include "TaskBlock.hpp"
#include "Processor.hpp"

void Router::HandleIncomingMessage(Message& msg)
{
    switch (msg.type)
    {
    case MessageTypeResponseCacheline:
        // Requested cacheline arrived
        tile->core.mmu.OnCachelineReceived(msg.requestId, msg.totalDelay, msg.cacheLine);
        break;

    case MessageTypeRequestL2:
        // Sender is requesting shared cache entry
        tile->core.mmu.FetchLocalL2(msg.sender, msg.addr);
        break;

    default:
        PrintLine("Invalid message type: " << (int)msg.type);
        assert(false);
        break;
    }
}

void Router::DispatchNext()
{
    if (msgQueue.size() == 0) return;

    // Dequeue next message
    queueLock.lock();
    Message& msg = msgQueue.front();
    msgQueue.pop_front();
    queueLock.unlock();

    // TODO: Simulate queueing delay for all waiting packets
    for (size_t i = 0; i < msgQueue.size(); ++i) {
        msg.totalDelay += GlobalConfig.QueuingDelay;
    }

    // Handle dispatch
    if (msg.receiver == tile->tileIdx || msg.IsBroadcast())
    {
        // The message is directed at this guy
        HandleIncomingMessage(msg);
    }
    else
    {
        if (msg.IsBroadcast())
        {
            // The message is a broadcast that goes to this guy and also to a bunch of other guys
            // NIY
        }
        else
        {
            // The Message is sent to another tile -> Route through
            RouteToNeighbor(msg);
        }
    }
}

void Router::RouteToNeighbor(const Message& msg)
{
    // Simulate transport delay
    int2 neighborIdx;

    Processor& processor = *tile->coreBlock->processor;
    int2 blockSize = processor.coreBlockSize;
    int2 coreOrigin = tile->coreBlock->ComputeCoreBlockOrigin();

    // TODO: Shortest path computation below (Hint: See Tile::IsBoundaryTile() for more information)

    if (!msg.IsReceiverTile())
    {
        // Message is sent to memory
        if (tile->IsBoundaryTile())
        {
            // Arrived at boundary -> Put in memory queue
            processor.globalMemory.Enqueue(msg);
            return;
        }

        // TODO: Shortest path to memory is always path to shortest boundary, i.e. min(x, y, width-x, width-y)
        neighborIdx = int2(0, 0);
    }
    else
    {
        // Message is sent to tile
        // TODO: Shortest path between two routers often allows for 2 choices at any point - Select one of the choices at random
        neighborIdx = int2(0, 0);
    }

    Router& neighbor = processor.GetTile(neighborIdx).router;
    neighbor.EnqueueMessage(msg);
}
#endif
