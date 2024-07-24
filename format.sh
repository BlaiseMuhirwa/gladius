
## Format header and c++ files
find src -type f -not -path "*cereal*" -iname *.hpp -o -iname *.cc | xargs clang-format -i  
find tests -iname *.hpp -o -iname *.cc | xargs clang-format -i 

## Format CMakeLists.txt files 
find . -type f -not -path "*dependencies*" -not -path "*data*" -not -path "*gladius_rust*" -not -path "*src/cereal*" -iname CMakeLists.txt | xargs cmake-format -i 
