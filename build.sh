#!/bin/bash

mkdir -p build 

## DCMAKE_EXPORT_COMPILE_COMMANDS=ON generates a compile_commands.json
## for clang-tidy. https://www.kdab.com/clang-tidy-part-1-modernize-source-code-using-c11c14/
## DCMAKE_CXX_COMPILER:FILEPATH sets the compiler path 
cd build && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++ ..
make 
cd ..