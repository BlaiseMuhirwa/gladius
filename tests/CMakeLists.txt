enable_testing()

include(GoogleTest)

add_executable(FortisMLPMnist e2e.cc)

target_link_libraries(FortisMLPMnist gtest gtest_main fortis)
