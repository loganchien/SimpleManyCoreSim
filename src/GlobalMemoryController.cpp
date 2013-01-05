#include "GlobalMemoryController.hpp"

#include "Address.hpp"
#include "Processor.hpp"
#include "Router.hpp"
#include "SimConfig.hpp"
#include "Tile.hpp"

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

    // Compute index of boundary router, closest to destination router
	Dim2 nearestRouterId;
	int gridLen = GlobalConfig.CoreGridLen*GlobalConfig.CoreBlockLen-1;
	if(std::min(request.sender.x,gridLen-request.sender.x)<std::min(request.sender.y,gridLen-request.sender.y)){
		//case: closer horizontally (start at closest x-value, correct y)
		nearestRouterId.y = request.sender.y;
		nearestRouterId.x = request.sender.x<gridLen-request.sender.x ? 0 : gridLen;
	}
	else{ // closer diagonally (start at correct x, closest y-value)
		nearestRouterId.x = request.sender.x;
		nearestRouterId.y = request.sender.y<gridLen-request.sender.y ? 0 : gridLen;
	}

    Router nearestRouter = processor->GetTile(nearestRouterId)->router; // TODO: fix?

    // Send out new response Message
    nearestRouter.EnqueueMessage(response);
}
