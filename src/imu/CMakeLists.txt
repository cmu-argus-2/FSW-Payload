

add_library(imu
    imu.cpp
)


target_link_libraries(imu
    PRIVATE core 
)