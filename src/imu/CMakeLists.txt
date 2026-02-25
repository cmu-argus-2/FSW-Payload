

add_library(imu
    imu_manager.cpp bmx160.cpp
)


target_link_libraries(imu
    PRIVATE core 
)