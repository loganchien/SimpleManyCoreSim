cmake_minimum_required (VERSION 2.6)

project (TileSim)

include_directories (
  external/boost
)

IF (WINDOWS)
  include_directories (
    external/msinttypes
  )
ENDIF (WINDOWS)

add_definitions (-DSMCSIM)

add_executable (TileSim
  external/armulator/src/ARM.cpp
  external/armulator/src/Thumb.cpp
  external/armulator/src/TileMMU.cpp
  external/armulator/src/elf_file.cpp
  external/armulator/src/swi_semihost.cpp
  src/Address.cpp
  src/Cache.cpp
  src/CoreBlock.cpp
  src/CPU.cpp
  src/GlobalMemoryController.cpp
  src/main.cpp
  src/Message.cpp
  src/MMU.cpp
  src/Processor.cpp
  src/Router.cpp
  src/SimConfig.cpp
  src/TaskBlock.cpp
  src/Task.cpp
  src/Thread.cpp
  src/Tile.cpp
)
