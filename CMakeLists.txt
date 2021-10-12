cmake_minimum_required(VERSION 3.10)
project(registration)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL 1.3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(registration src/main.cpp src/config.cpp src/csv_parser.cpp src/align.cpp)
target_link_libraries(registration ${PCL_LIBRARIES} yaml-cpp)
target_include_directories(registration PUBLIC include/)