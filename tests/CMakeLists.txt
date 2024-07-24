enable_testing()

include(GoogleTest)

add_executable(gladius_mlp_mnist e2e.cc)

target_link_libraries(gladius_mlp_mnist GTest::gtest_main gladius)
gtest_discover_tests(gladius_mlp_mnist)
