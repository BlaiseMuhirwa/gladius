mkdir -p build 
cd build && cmake -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++ ..
make 
cd ..