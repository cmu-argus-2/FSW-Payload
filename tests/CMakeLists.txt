# Create an executable for the tests
add_executable(PAYLOAD_TESTS main.cpp
    test_thread_pool.cpp
    test_errors.cpp
    navigation/test_quaternion.cpp
    test_tilepack_dh.cpp
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

add_executable(TEST_CAMERA test_camera.cpp)
target_link_libraries(TEST_CAMERA
    PRIVATE
    PAYLOAD_LIB
)
