
find src -type f -not -path "*cereal*" -iname *.hpp -o -iname *.cc | xargs clang-format -i  