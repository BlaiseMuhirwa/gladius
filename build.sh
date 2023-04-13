#!/bin/bash

mkdir -p build 

TESTING=OFF

if [ $1 = "tests" ]
then 
    TESTING=ON
fi


## DCMAKE_EXPORT_COMPILE_COMMANDS=ON generates a compile_commands.json
## for clangd. https://www.kdab.com/clang-tidy-part-1-modernize-source-code-using-c11c14/
## DCMAKE_CXX_COMPILER:FILEPATH sets the compiler path 
## DCMAKE_EXPORT_COMPILE_COMMANDS=OFF since clangd seems to consume so much 
## memory and cpu time 
cd build && cmake -DBUILD_TESTS=${TESTING} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER:FILEPATH=/usr/local/opt/llvm/bin/clang++ ..
make 
cd ..