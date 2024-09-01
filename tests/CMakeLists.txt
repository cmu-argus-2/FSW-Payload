# Create an executable for the region classifier test
add_executable(test_region_classifier vision/test_region_classifier.cpp)

# Link the test executable with Google Test and the PAYLOAD library
target_link_libraries(test_region_classifier 
    PRIVATE 
        GTest::gtest 
        GTest::gtest_main 
        PAYLOAD_LIB
        ${TORCH_LIBRARIES}
)

# Register the region classifier test with CTest
add_test(
    NAME RegionClassifierTest
    COMMAND test_region_classifier
)
