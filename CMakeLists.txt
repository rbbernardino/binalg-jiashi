cmake_minimum_required(VERSION 2.8)

set (PROJ_NAME cppbinarize)
project(${PROJ_NAME})

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package( OpenCV REQUIRED )

set(HEADERS utils.h binalgorithms.h)
set(SOURCES cppbinarize.cpp utils.cpp binalgorithms.cpp args.hxx)

include_directories("doxa")


# set as static build
SET(BUILD_SHARED_LIBRARIES OFF)

add_executable( cppbinarize ${SOURCES} ${HEADERS} )

target_link_libraries( cppbinarize ${OpenCV_LIBS} )
