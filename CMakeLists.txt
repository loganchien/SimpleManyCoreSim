cmake_minimum_required (VERSION 2.6)

project (TileSim)

include_directories (
  external/armulator/src
  external/boost
)

IF (WINDOWS)
  include_directories (
    external/msinttypes
  )
ENDIF (WINDOWS)

add_definitions (-DNDEBUG)

add_executable (TileSim
  external/armulator/src/ARM.cpp
  external/armulator/src/ArmulatorCPU.cpp
  external/armulator/src/ArmulatorMMU.cpp
  external/armulator/src/Thumb.cpp
  external/armulator/src/elf_file.cpp
  external/armulator/src/swi_semihost.cpp
  external/boost/libs/filesystem/src/path_traits.cpp
  external/boost/libs/filesystem/src/portability.cpp
  external/boost/libs/filesystem/src/unique_path.cpp
  external/boost/libs/filesystem/src/utf8_codecvt_facet.cpp
  external/boost/libs/filesystem/src/windows_file_codecvt.cpp
  external/boost/libs/filesystem/src/path.cpp
  external/boost/libs/filesystem/src/operations.cpp
  external/boost/libs/filesystem/src/codecvt_error_category.cpp
  external/boost/libs/system/src/error_code.cpp
  src/Address.cpp
  src/Cache.cpp
  src/Core.cpp
  src/CoreBlock.cpp
  src/GlobalMemoryController.cpp
  src/MMU.cpp
  src/Message.cpp
  src/Processor.cpp
  src/Router.cpp
  src/SimConfig.cpp
  src/Task.cpp
  src/TaskBlock.cpp
  src/Thread.cpp
  src/Tile.cpp
  src/main.cpp
)

set_target_properties(TileSim PROPERTIES COMPILE_FLAGS -DSMCSIM)

#add_executable (armulator
#  external/armulator/src/ARM.cpp
#  external/armulator/src/Thumb.cpp
#  external/armulator/src/MMU.cpp
#  external/armulator/src/elf_file.cpp
#  external/armulator/src/main.cpp
#  external/armulator/src/swi_semihost.cpp
#)
