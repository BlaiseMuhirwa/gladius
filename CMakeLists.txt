cmake_minimum_required(VERSION 3.18)


project(fortis_nn LANGUAGES CXX)

set(CMAKE_C_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)

## set compiler flags 
## TODO: Add -lbenchmark compiler flag and the corresponding
## benchmark submodule https://github.com/google/benchmark
set(CMAKE_CXX_FLAGS " -std=c++17 -O3")

include_directories(".")
include_directories(${CMAKE_BINARY_DIR})

## This requires having a CMakeLists.txt file in the subdirectory 
add_subdirectory(src)
add_subdirectory(src/comp_graph)
add_subdirectory(src/comp_graph/vertices)

## Add dependencies 
set(SKIP_PERFORMANCE_COMPARISON ON)
add_subdirectory(dependencies/cereal EXCLUDE_FROM_ALL)

include_directories(dependencies)
include_directories(src)
include_directories(src/cereal)
include_directories(src/comp_graph)
include_directories(src/comp_graph/vertices)


set(FORTIS_SOURCES
    ${PROJECT_SOURCE_DIR}/src/trainer.cc 
    ${PROJECT_SOURCE_DIR}/src/model.cc
    ${PROJECT_SOURCE_DIR}/src/attention.cc
)

add_library(fortis STATIC ${FORTIS_SOURCES})

target_include_directories(
    fortis
    PRIVATE 
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/comp_graph
    ${PROJECT_SOURCE_DIR}/src/operations
)
set_target_properties(fortis PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fortis PUBLIC)

add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cc)

message("===================================================")
message("\tProject Directory = ${PROJECT_SOURCE_DIR}")
message("\tC++ Compiler = ${CMAKE_CXX_COMPILER}")
message("\tCMake Source Dir = ${CMAKE_CURRENT_SOURCE_DIR}")
message("\tCMake Binary Dir = ${CMAKE_BINARY_DIR}")
message("\tSource Files = ")
foreach(SOURCE_FILE IN LISTS FORTIS_SOURCES)
    message("\t ${SOURCE_FILE}")
endforeach(SOURCE_FILE IN LISTS FORTIS_SOURCES)