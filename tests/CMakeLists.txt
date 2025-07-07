# Create an executable for the tests
add_executable(PAYLOAD_TESTS main.cpp
    test_thread_pool.cpp
    test_errors.cpp
    navigation/test_quaternion.cpp
    )

# Link the test executable with Google Test and the PAYLOAD library
target_link_libraries(PAYLOAD_TESTS
    PRIVATE 
        GTest::gtest 
        GTest::gtest_main 
        PAYLOAD_LIB
)

add_test(NAME PAYLOAD_TESTS COMMAND PAYLOAD_TESTS)

# Create an executable for inference test
add_executable(RUN_INFERENCE run_inference.cpp)

target_link_libraries(RUN_INFERENCE
    PRIVATE
    PAYLOAD_LIB
)
