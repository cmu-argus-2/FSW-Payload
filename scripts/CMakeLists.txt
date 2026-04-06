# # Create an executable for inference test
if(CUDA_ENABLED)
    add_executable(RUN_INFERENCE run_inference.cpp)
    target_link_libraries(RUN_INFERENCE
        PRIVATE
        PAYLOAD_LIB
    )
endif()

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

add_executable(RUN_CAMERA_INFERENCE run_camera_inference.cpp)
target_link_libraries(RUN_CAMERA_INFERENCE
    PRIVATE
    PAYLOAD_LIB
)

add_executable(RUN_PREFILTER run_prefiltering.cpp)
target_link_libraries(RUN_PREFILTER
    PRIVATE
    vision
    ${OpenCV_LIBS}
)

add_executable(RUN_BATCH_OPT navigation/run_batch_opt.cpp)
target_link_libraries(RUN_BATCH_OPT
    PRIVATE
    spdlog::spdlog
    PAYLOAD_LIB
    HDF5::HDF5           
    HighFive::HighFive
    Ceres::ceres
    Eigen3::Eigen
)


add_executable(yolo_detector
    yolo_detector.cpp
)
target_link_libraries(yolo_detector
    PRIVATE
    PAYLOAD_LIB
    spdlog::spdlog
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