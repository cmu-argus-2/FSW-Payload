set(PAYLOAD_TEST_SOURCES
    main.cpp
    test_thread_pool.cpp
    test_errors.cpp
    navigation/test_quaternion.cpp
    test_tilepack_dh.cpp
    test_imu.cpp
    test_dataset.cpp
    test_commands.cpp
)

if(CUDA_ENABLED)
    list(APPEND PAYLOAD_TEST_SOURCES test_inference.cpp)
endif()
# Create an executable for the tests
add_executable(PAYLOAD_TESTS ${PAYLOAD_TEST_SOURCES})

# Link the test executable with Google Test and the PAYLOAD library
target_link_libraries(PAYLOAD_TESTS
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        PAYLOAD_LIB
)

if(CUDA_ENABLED)
    target_include_directories(PAYLOAD_TESTS PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

add_test(NAME PAYLOAD_TESTS COMMAND PAYLOAD_TESTS)