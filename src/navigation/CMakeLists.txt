find_package(Eigen3 3.3 REQUIRED)

add_library(navigation
    od.cpp
    od_alg.cpp
    od_measurements.cpp
    batch_optimization.cpp
    trajectory_initializer.cpp
    pose_dynamics.cpp
    quaternion.cpp
    utils.cpp
)

target_link_libraries(navigation
        PUBLIC  casadi::casadi
        PRIVATE Eigen3::Eigen HDF5::HDF5 spice opencv_calib3d
                nlohmann_json::nlohmann_json)
