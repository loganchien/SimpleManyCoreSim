
#include "Message.hpp"

#include <mutex>
#include <deque>

struct Tile;
struct Processor;

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
    void HandleIncomingMessage(Message& msg);

    /**
     * Sends the message with highest priority to it's next target (either this tile's MMU, a neighboring router, the global MMU (, or the entire core block)).
     * Called by Processor.
     */
    void DispatchNext();

    /// Send message to next Tile on the shortest path to target
    void RouteToNeighbor(const Message& msg);

    /// Enqueue message on this router
    void EnqueueMessage(const Message& msg)
    {
        // Add to queue
        ++simTotalPacketsReceived;

        // FIXME: Add the delay
        //msg.totalDelay += GlobalConfig.Route1Delay;

        queueLock.lock();
        msgQueue.push_back(msg);
        queueLock.unlock();
    }
};
