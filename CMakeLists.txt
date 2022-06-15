cmake_minimum_required(VERSION 3.16)
project(registration)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found! You need OpenMP for speedup on multicore CPUs!")
endif()

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(registration src/main.cpp src/config.cpp src/csv_parser.cpp src/align.cpp src/common.cpp src/filter.cpp src/analysis.cpp src/feature_analysis.cpp src/downsample.cpp src/utils.cpp src/metric.cpp src/weights.cpp src/gror/gror_pre.cpp)
target_link_libraries(registration ${PCL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)
target_include_directories(registration PUBLIC include/)

enable_testing()
add_subdirectory(tests)

add_subdirectory(experiments)