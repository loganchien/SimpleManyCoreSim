#include "Router.hpp"

#include "Core.hpp"
#include "CoreBlock.hpp"
#include "Debug.hpp"
#include "Processor.hpp"
#include "SimConfig.hpp"
#include "Tile.hpp"

#include <algorithm>
#include <assert.h>

using namespace smcsim;

/// Initialize a new Router object
void Router::InitRouter(Tile* tile)
{
    this->tile = tile;
}

// #################### Handle messages #######################################

/// Called when a Message directed at this tile is dispatched
void Router::HandleIncomingMessage(Message& msg)
{
    switch (msg.type)
    {
    case MessageTypeResponseCacheline:
        // Requested cacheline arrived
        tile->mmu.OnCachelineReceived(msg.requestId, msg.totalDelay, msg.cacheLine);
        break;

    case MessageTypeRequestL2:
        // Sender is requesting shared cache entry
        tile->mmu.FetchLocalL2(msg.sender,msg.requestId, msg.totalDelay, msg.addr);
        break;

    default:
        PrintLine("Invalid message type: " << (int)msg.type);
        assert(false);
        break;
    }
}

/// Sends the message with highest priority to it's next target (either this
/// tile's MMU, a neighboring router, the global MMU (, or the entire core
/// block)).  Called by Processor.
void Router::DispatchNext()
{
    if (msgQueue.size() == 0) return;

    // Simulate queueing delay for all waiting packets
	for(int i=0;i<msgQueue.size();i++){
		msgQueue.at(i).totalDelay+= GlobalConfig.QueuingDelay;
	}
    // Dequeue next message
    Message& msg = msgQueue.front();
    msgQueue.pop_front();

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
            // The message is a broadcast that goes to this guy and also to a
            // bunch of other guys
            //NIY
        }
        else
        {
            // The Message is sent to another tile -> Route through
            RouteToNeighbor(msg);
        }
    }
}


// #################### Transport messages ####################################

/// Send message to next Tile on the shortest path to target
void Router::RouteToNeighbor(Message& msg)
{
    // Simulate transport delay
    Dim2 neighborIdx;

    Processor& processor = *tile->coreBlock->processor;
    Dim2 blockSize = processor.coreBlockSize;
    Dim2 coreOrigin = tile->coreBlock->ComputeCoreBlockOrigin();

    // Shortest path computation below (Hint: See Tile::IsBoundaryTile()
    // for more information)

    if (!msg.IsReceiverTile())
    {
        // Message is sent to memory
        if (tile->IsBoundaryTile())
        {
            // Arrived at boundary -> Put in memory queue
            processor.gMemController.EnqueueRequest(msg);
            return;
        }

        // Shortest path to memory is always path to shortest boundary,
        // i.e. min(x, y, width-x, width-y)
        if (std::min(tile->tileIdx.x, GlobalConfig.TotalCoreLength() - tile->tileIdx.x) <
            std::min(tile->tileIdx.y, GlobalConfig.TotalCoreLength() - tile->tileIdx.y))
        {
            //move horizontally to border
            if (tile->tileIdx.x < GlobalConfig.TotalCoreLength() - tile->tileIdx.x)
                neighborIdx = Dim2(tile->tileIdx.y, tile->tileIdx.x-1);
            else
                neighborIdx = Dim2(tile->tileIdx.y, tile->tileIdx.x+1);
        }
        else{	//move vertically
            if (tile->tileIdx.y < GlobalConfig.TotalCoreLength() - tile->tileIdx.y)
                neighborIdx = Dim2(tile->tileIdx.y-1, tile->tileIdx.x);
            else
                neighborIdx = Dim2(tile->tileIdx.y+1, tile->tileIdx.x);
        }
    }
    else
    {
        // Message is sent to tile
        // Shortest path between two routers often allows for 2 choices
        // at any point - Select one of the choices at random
        int r = rand() % 2;
        int next;
        if(r==0){ //prefer horizontal movement
            if(msg.receiver.x != tile->tileIdx.x){
                next = (msg.receiver.x > tile->tileIdx.x) ? tile->tileIdx.x+1 : tile->tileIdx.x-1 ;
                neighborIdx = Dim2(tile->tileIdx.y, next);
            }
            else { //already on correct x index, move vertical!
                next = (msg.receiver.y > tile->tileIdx.y) ? tile->tileIdx.y+1 : tile->tileIdx.y-1 ;
                neighborIdx = Dim2(next, tile->tileIdx.x);
            }
        }
        else {	//perfer vertical movement
            if(msg.receiver.y != tile->tileIdx.y){
                next = (msg.receiver.y > tile->tileIdx.y) ? tile->tileIdx.y+1 : tile->tileIdx.y-1 ;
                neighborIdx = Dim2(next, tile->tileIdx.x);
            }
            else{
                next = (msg.receiver.x > tile->tileIdx.x) ? tile->tileIdx.x+1 : tile->tileIdx.x-1 ;
                neighborIdx = Dim2(tile->tileIdx.y, next);
            }
        }
    }

    Router& neighbor = processor.GetTile(neighborIdx)->router;
    neighbor.EnqueueMessage(msg);
}

/// Enqueue message on this router
void Router::EnqueueMessage(Message& msg)
{
    // Add to queue
    ++simTotalPacketsReceived;
    msg.totalDelay += GlobalConfig.Route1Delay;

    msgQueue.push_back(msg);
}
