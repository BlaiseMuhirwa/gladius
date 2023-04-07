enable_testing()

include(GoogleTest)

add_executable(FortisMLPMnist e2e.cc)

target_link_libraries(FortisMLPMnist GTest::gtest_main fortis)
gtest_discover_tests(FortisMLPMnist)
