#include "GlobalMemoryController.hpp"

#include "Address.hpp"
#include "Processor.hpp"
#include "Router.hpp"
#include "SimConfig.hpp"
#include "Tile.hpp"

#include <assert.h>
#include <string.h>

using namespace smcsim;

GlobalMemoryController::GlobalMemoryController()
    : processor(NULL),
      text(NULL), textVMA(0), textSize(0),
      data(NULL), dataVMA(0), dataSize(0),
      rodata(NULL), rodataVMA(0), rodataSize(0),
      bss(NULL), bssVMA(0), bssSize(0),
      heap(NULL), heapVMA(0), heapSize(0),
      stackBegin(NULL), stackBeginVMA(0), stackBeginSize(0),
      stack(NULL), stackVMA(0)
{
}

void GlobalMemoryController::InitGMemController(Processor* processor_)
{
    processor = processor_;
}

void GlobalMemoryController::Reset()
{
    delete [] text;
    delete [] data;
    delete [] rodata;
    delete [] bss;
    delete [] heap;
    delete [] stackBegin;

    text = NULL;
    textVMA = 0;
    textSize = 0;

    data = NULL;
    dataVMA = 0;
    dataSize = 0;

    rodata = NULL;
    rodataVMA = 0;
    rodataSize = 0;

    bss = NULL;
    bssVMA = 0;
    bssSize = 0;

    heap = NULL;
    heapVMA = 0;
    heapSize = 0;

    stackBegin = NULL;
    stackBeginVMA = 0;
    stackBeginSize = 0;

    stack = NULL;
    stackVMA = 0;
}

void GlobalMemoryController::LoadExecutable(elf_file* file)
{
}

void GlobalMemoryController::StoreCoreDump()
{
    assert(0 && "Not implemented");
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
    memcpy(&*response.cacheLine.bytes.begin(),
           GetMemory(request.addr.raw, GlobalConfig.CacheLineSize),
           GlobalConfig.CacheLineSize);

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

uint8_t* GlobalMemoryController::GetMemory(uint32_t addr, uint32_t size)
{
#define MEMORY_RANGE(NAME) \
    if (addr >= (NAME##VMA) && addr < ((NAME##VMA) + (NAME##Size))) \
    { \
        assert(addr + size <= ((NAME##VMA) + (NAME##Size))); \
        return (NAME + (addr - (NAME##VMA))); \
    }

    MEMORY_RANGE(text);
    MEMORY_RANGE(data);
    MEMORY_RANGE(rodata);
    MEMORY_RANGE(bss);
    MEMORY_RANGE(heap);
    MEMORY_RANGE(stackBegin);

#undef MEMORY_RANGE

    return NULL;
}
