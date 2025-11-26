find_package (Eigen3 3.3 REQUIRED)

add_library(navigation
    od.cpp
    od_alg.cpp
    batch_optimization.cpp
    quaternion.cpp
    utils.cpp
)

target_link_libraries(navigation 
        PRIVATE Eigen3::Eigen HighFive::HighFive Ceres::ceres HDF5::HDF5)

# target_link_libraries(navigation <dependencies>)
