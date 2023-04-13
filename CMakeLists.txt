cmake_minimum_required(VERSION 3.18)

project(fortis_nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_COMPILER /usr/local/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)

# Compile with OpenMP for parallel optimizations
find_package(OpenMP)
if(NOT OpenMP_FOUND)
  message(STATUS "OpenMP not found. Attemting to build it from scratch.")

  set(LLVM_OPENMP_VERSION "13.0.1")
  set(LLVM_OPENMP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/openmp")

  if(NOT EXISTS ${LLVM_OPENMP_SOURCE_DIR})
    message(
      STATUS
        "Downloading openmp-${LLVM_OPENMP_VERSION} into ${LLVM_OPENMP_SOURCE_DIR}"
    )
    set(LLVM_OPENMP_DOWNLOAD_URL
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_OPENMP_VERSION}/openmp-${LLVM_OPENMP_VERSION}.src.tar.xz"
    )
    file(
      DOWNLOAD ${LLVM_OPENMP_DOWNLOAD_URL} # URL
      "${CMAKE_CURRENT_BINARY_DIR}/openmp-${LLVM_OPENMP_VERSION}.tar.bz2" # FILE
    )
    file(ARCHIVE_EXTRACT INPUT
         "${CMAKE_CURRENT_BINARY_DIR}/openmp-${LLVM_OPENMP_VERSION}.tar.bz2"
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/openmp-${LLVM_OPENMP_VERSION}.src"
         "${LLVM_OPENMP_SOURCE_DIR}")
  endif()
  set(OPENMP_STANDALONE_BUILD ON)
  add_subdirectory(${LLVM_OPENMP_SOURCE_DIR} EXCLUDE_FROM_ALL)
  add_library(OpenMP::OpenMP_CXX ALIAS omp)

  message(STATUS "Installed OpenMP: ${LLVM_OPENMP_SOURCE_DIR}")

else()
  message(STATUS "Compiling with OpenMP Optimizations")
  message(STATUS "OpenMP Version: ${OpenMP_VERSION}")
  message(STATUS "OpenMP directory: ${OpenMP_CXX_INCLUDE_DIR}")
endif(NOT OpenMP_FOUND)

option(BUILD_TESTS "Builds unit and integration tests" OFF)

# set compiler flags TODO: Add -lbenchmark compiler flag and the corresponding
# benchmark submodule https://github.com/google/benchmark
set(CMAKE_CXX_FLAGS " -std=c++20 -fopenmp -O3")

include_directories(".")
include_directories(${CMAKE_BINARY_DIR})

# Where to find header files
# include_directories(src/comp_graph)
# include_directories(src/comp_graph/vertices)
# include_directories(dependencies)
# include_directories(src)

# This requires having a CMakeLists.txt file in the subdirectory
add_subdirectory(src)

if (BUILD_TESTS)
  add_subdirectory(tests)
  message(STATUS "Building unit and integration tests")
endif()

add_subdirectory(src/comp_graph)
add_subdirectory(src/trainers)
add_subdirectory(src/params)
add_subdirectory(src/comp_graph/vertices)

# Add dependencies
set(SKIP_PERFORMANCE_COMPARISON ON)
add_subdirectory(dependencies/cereal EXCLUDE_FROM_ALL)
add_subdirectory(dependencies/googletest EXCLUDE_FROM_ALL)

# include(FetchContent) FetchContent_Declare( googletest URL
# https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
# )

set(FORTIS_SOURCES
    ${PROJECT_SOURCE_DIR}/src/trainers/trainer.cc
    ${PROJECT_SOURCE_DIR}/src/model.cc ${PROJECT_SOURCE_DIR}/src/attention.cc)

add_library(fortis STATIC ${FORTIS_SOURCES})

target_include_directories(
  fortis PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/comp_graph
                 ${PROJECT_SOURCE_DIR}/src/operations)
set_target_properties(fortis PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fortis PUBLIC OpenMP::OpenMP_CXX cereal::cereal)

message("===================================================")
message(STATUS "\tProject Directory = ${PROJECT_SOURCE_DIR}")
message(STATUS "\tC++ Compiler = ${CMAKE_CXX_COMPILER}")
message(STATUS "\tCMake Source Dir = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "\tCMake Binary Dir = ${CMAKE_BINARY_DIR}")
message(STATUS "\tSource Files = ")
foreach(SOURCE_FILE IN LISTS FORTIS_SOURCES)
  message(STATUS "\t ${SOURCE_FILE}")
endforeach(SOURCE_FILE IN LISTS FORTIS_SOURCES)
