# Create an executable for the region classifier test
add_executable(PAYLOAD_TESTS main.cpp
    test_thread_pool.cpp
    vision/test_region_classifier.cpp
    navigation/test_quaternion.cpp
    )

# Link the test executable with Google Test and the PAYLOAD library
target_link_libraries(PAYLOAD_TESTS
    PRIVATE 
        GTest::gtest 
        GTest::gtest_main 
        PAYLOAD_LIB
        ${TORCH_LIBRARIES}
)



add_test(NAME PAYLOAD_TESTS COMMAND PAYLOAD_TESTS)
