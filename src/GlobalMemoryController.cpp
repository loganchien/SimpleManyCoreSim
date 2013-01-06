#include "GlobalMemoryController.hpp"

#include "Address.hpp"
#include "Debug.hpp"
#include "Processor.hpp"
#include "Router.hpp"
#include "SimConfig.hpp"
#include "Tile.hpp"

#include "ArmulatorError.h"
#include "elf_file.h"

#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

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
      stack(NULL), stackVMA(0), entryPointVMA(0)
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

    entryPointVMA = 0;
}

void GlobalMemoryController::LoadExecutable(std::ifstream& stream)
{
    elf_file elf;
    elf.get_info(stream);

    for (int i = 0; i < elf.elf_header.e_shnum; i++)
    {
        int name_offset = elf.sec_header[i].sh_name;
        const char* name = &elf.sec_name[name_offset];

        int sh_offset = elf.sec_header[i].sh_offset;
        int sh_vma = elf.FileOff2VMA(sh_offset);
        int sh_size = elf.sec_header[i].sh_size;

        if (strcmp(name, ".text") == 0)
        {
            text = new uint8_t[sh_size];
            textVMA = sh_vma;
            textSize = sh_size;
            stream.seekg(sh_offset, std::ios_base::beg);
            stream.read(reinterpret_cast<char*>(text), sh_size);
        }
        else if (strcmp(name, ".data") == 0)
        {
            data = new uint8_t[sh_size];
            dataVMA = sh_vma;
            dataSize = sh_size;
            stream.seekg(sh_offset, std::ios_base::beg);
            stream.read(reinterpret_cast<char*>(data), sh_size);
        }
        else if (strcmp(name, ".rodata") == 0)
        {
            rodata = new uint8_t[sh_size];
            rodataVMA = sh_vma;
            rodataSize = sh_size;
            stream.seekg(sh_offset, std::ios_base::beg);
            stream.read(reinterpret_cast<char*>(rodata), sh_size);
        }
        else if (strcmp(name, ".bss") == 0)
        {
            bss = new uint8_t[sh_size];
            bssVMA = sh_vma;
            bssSize = sh_size;
        }
    }

    // Allocate the heap
    heapSize = GlobalConfig.HeapSize;
    heapVMA = 0x80000000u;
    heap = new uint8_t[heapSize];

    // Allocate the stack for multiple cores
    stackBeginSize = GlobalConfig.StackSize *
                     GlobalConfig.CoreBlockSize().Area() *
                     GlobalConfig.CoreGridSize().Area();
    stackBeginVMA = heapVMA + heapSize;
    stackBegin = new uint8_t[stackBeginSize];
    stackVMA = stackBeginVMA + stackBeginSize;
    stack = stackBegin + stackBeginSize;

    PrintLine("Address Space: textVMA=" << textVMA);
    PrintLine("Address Space: dataVMA=" << dataVMA);
    PrintLine("Address Space: rodataVMA=" << rodataVMA);
    PrintLine("Address Space: bssVMA=" << bssVMA);
    PrintLine("Address Space: stackBeginVMA=" << stackBeginVMA);

    // Load Entry Point
    entryPointVMA = elf.getEntryPoint();
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

/// Interface to access the memory
uint8_t GlobalMemoryController::LoadByte(uint32_t addr)
{
    return *GetMemory(addr, 1);
}

uint16_t GlobalMemoryController::LoadHalfWord(uint32_t addr)
{
    return *reinterpret_cast<uint16_t*>(GetMemory(addr, 2));
}

uint32_t GlobalMemoryController::LoadWord(uint32_t addr)
{
    return *reinterpret_cast<uint32_t*>(GetMemory(addr, 4));
}

void GlobalMemoryController::StoreByte(uint32_t addr, uint8_t byte)
{
    *GetMemory(addr, 1) = byte;
}

void GlobalMemoryController::StoreHalfWord(uint32_t addr, uint16_t halfword)
{
    *reinterpret_cast<uint16_t*>(GetMemory(addr, 2)) = halfword;
}

void GlobalMemoryController::StoreWord(uint32_t addr, uint32_t word)
{
    *reinterpret_cast<uint32_t*>(GetMemory(addr, 4)) = word;
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

    std::stringstream ss;
    ss << "Segmentation fault: 0x" << std::hex << addr;
    throw UnexpectInst(ss.str());
}
