##### INFERENCE

if(CUDA_ENABLED)
    add_executable(RUN_INFERENCE run_inference.cpp)
    target_link_libraries(RUN_INFERENCE
        PRIVATE
        PAYLOAD_LIB
    )
endif()

##### BATCH OPTIMIZATION

add_executable(RUN_BATCH_OPT run_batch_opt.cpp)
target_link_libraries(RUN_BATCH_OPT
    PRIVATE
    spdlog::spdlog
    PAYLOAD_LIB
    HDF5::HDF5           
    HighFive::HighFive
    Ceres::ceres
    Eigen3::Eigen
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

##### DATASETS

add_executable(run_dataset
    run_dataset.cpp
)
target_link_libraries(run_dataset
    PRIVATE
    spdlog::spdlog
    PAYLOAD_LIB
)

##### CAMERAS

add_executable(RUN_CAMERA run_camera.cpp)
target_link_libraries(RUN_CAMERA
    PRIVATE
    PAYLOAD_LIB
)

add_executable(GEN_TILEPACK_RAW tilepack_test/generate_tilepack_raw.cpp)
target_link_libraries(GEN_TILEPACK_RAW
    PRIVATE
    tilepack
    core
    vision
)

##### PREFILTERING

add_executable(RUN_PREFILTER run_prefiltering.cpp)
target_link_libraries(RUN_PREFILTER
    PRIVATE
    vision
    ${OpenCV_LIBS}
)

##### IMU

add_executable(imu_run run_imu.cpp)
target_link_libraries(imu_run PRIVATE PAYLOAD_LIB)