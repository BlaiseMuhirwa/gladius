cmake_minimum_required(VERSION 3.20)


project(fortis_nn LANGUAGES CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(".")
add_subdirectory(src)


## set compiler flags 
set(CMAKE_CXX_FLAGS "-O3 -W")


set(FORTIS_SOURCES
    ${PROJECT_SOURCE_DIR}/src/trainer.cc 
    ${PROJECT_SOURCE_DIR}/src/model.cc
    ${PROJECT_SOURCE_DIR}/src/attention.cc
    ${PROJECT_SOURCE_DIR}/src/comp_graph/vertex.cc 
)

## where to find header files
# include_directories(${PROJECT_SOURCE_DIR}/src)
# include_directories(${PROJECT_SOURCE_DIR}/src/comp_graph)
# include_directories(${PROJECT_SOURCE_DIR}/src/operators)

add_library(fortis STATIC ${FORTIS_SOURCES})
target_include_directories(
    fortis
    PUBLIC 
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/comp_graph
    ${PROJECT_SOURCE_DIR}/src/operators
)
set_target_properties(fortis PROPERTIES POSITION_INDEPENDENT_CODE ON)

message("===================================================")
message("\tProject Directory = ${PROJECT_SOURCE_DIR}")
message("\tC++ Compiler = ${CMAKE_CXX_COMPILER}")
message("\tSource Files = ")
foreach(SOURCE_FILE IN LISTS FORTIS_SOURCES)
    message("\t ${SOURCE_FILE}")
endforeach(SOURCE_FILE IN LISTS FORTIS_SOURCES)