#ifndef ROUTER_HPP
#define ROUTER_HPP

#include "Message.hpp"

#include <deque>

namespace smcsim {

class Tile;

/// A router receives packets from and sends them to immediate neighbors or
/// it's own core
class Router
{
public:
    Tile* tile;
    std::deque<Message> msgQueue;

    /// Sim stats: How many packets went through this router
    int simTotalPacketsReceived;

public:
    /// Initialize a new Router object
    void InitRouter(Tile* tile);

    // #################### Handle messages ###################################

    /// Called when a Message directed at this tile is dispatched
    void HandleIncomingMessage(Message& msg);

    /// Sends the message with highest priority to it's next target (either this
    /// tile's MMU, a neighboring router, the global MMU (, or the entire core
    /// block)).  Called by Processor.
    void DispatchNext();


    // #################### Transport messages ################################

    /// Send message to next Tile on the shortest path to target
    void RouteToNeighbor(Message& msg);

    /// Enqueue message on this router
    void EnqueueMessage(Message& msg);
};

} // end namespace smcsim

#endif // ROUTER_HPP
