
#include "Message.hpp"

#include <mutex>
#include <deque>

/// A router receives packets from and sends them to immediate neighbors or it's own core
struct Router
{
    Tile* tile;
    std::mutex queueLock;
    std::deque<Message> msgQueue;

    /// Sim stats: How many packets went through this router
    int simTotalPacketsReceived;
    
    /// Initialize a new Router object
    void InitRouter(Tile* tile)
    {
        this->tile = tile;
    }

    // ############################################## Handle messages ##############################################

    /// Called when a Message directed at this tile is dispatched
    void HandleIncomingMessage(Message& msg)
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

    /**
     * Sends the message with highest priority to it's next target (either this tile's MMU, a neighboring router, the global MMU (, or the entire core block)).
     * Called by Processor.
     */
    void DispatchNext()
    {
        if (msgQueue.size() == 0) return;

        // TODO: Simulate queueing delay for all waiting packets
        msg.totalDelay += GlobalConfig.QueuingDelay;

        // Dequeue next message
        queueLock.lock();
        Message& msg = msgQueue.front();
        msgQueue.pop_front();
        queueLock.unlock();

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
    
    
    // ############################################## Transport messages ##############################################

    /// Send message to next Tile on the shortest path to target
    void RouteToNeighbor(const Message& msg)
    {
        // Simulate transport delay
        int2 neighborIdx;
  
        Processor& processor = *tile->coreBlock->processor;
        int2 blockSize = processor->coreBlockSize;
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
            neighborIdx = ;
        }
        else
        {
            // Message is sent to tile
            // TODO: Shortest path between two routers often allows for 2 choices at any point - Select one of the choices at random
            neighborIdx = ;
        }
         
        Router& neighbor = processor.GetTile(neighborIdx).router;
        neighbor.EnqueueMessage(msg);
    }

    /// Enqueue message on this router
    void EnqueueMessage(const Message& msg)
    {
        // Add to queue
        ++simTotalPacketsReceived;
        msg.totalDelay += GlobalConfig.Route1Delay;

        queueLock.lock();
        msgQueue.push_back(msg);
        queueLock.unlock();
    }
};