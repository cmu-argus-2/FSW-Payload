# Create an executable for the tests
add_executable(PAYLOAD_TESTS main.cpp
    test_thread_pool.cpp
    test_errors.cpp
    navigation/test_quaternion.cpp
    test_tilepack_dh.cpp
    test_imu.cpp
    test_dataset.cpp
)

# Link the test executable with Google Test and the PAYLOAD library
target_link_libraries(PAYLOAD_TESTS
    PRIVATE 
        GTest::gtest 
        GTest::gtest_main 
        PAYLOAD_LIB
)

add_test(NAME PAYLOAD_TESTS COMMAND PAYLOAD_TESTS)

add_executable(GEN_TILEPACK_RAW tilepack_test/generate_tilepack_raw.cpp)
target_link_libraries(GEN_TILEPACK_RAW
    PRIVATE
    tilepack
    core
    vision
)

# temporary executable to test jacobians
add_executable(test_jacobians
    navigation/test_jacobians.cpp
)
target_link_libraries(test_jacobians
    PRIVATE
    spdlog::spdlog
    PAYLOAD_LIB
    HDF5::HDF5           
    HighFive::HighFive
    Ceres::ceres
    Eigen3::Eigen
)
