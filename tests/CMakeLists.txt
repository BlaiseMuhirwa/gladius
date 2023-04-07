enable_testing()

include(GoogleTest)

add_executable(fortis_mlp_mnist e2e.cc)

target_link_libraries(fortis_mlp_mnist GTest::gtest_main fortis)
gtest_discover_tests(fortis_mlp_mnist)
