cmake_minimum_required(VERSION 3.16)
project(registration)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found! You need OpenMP for speedup on multicore CPUs!")
endif()

find_package(PCL 1.3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(registration src/main.cpp src/config.cpp src/csv_parser.cpp src/align.cpp)
target_precompile_headers(registration PRIVATE include/common.h)
target_link_libraries(registration ${PCL_LIBRARIES} yaml-cpp)
target_include_directories(registration PUBLIC include/)