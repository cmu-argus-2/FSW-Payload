

add_library(vision
    camera_manager.cpp
    camera.cpp
    frame.cpp
    rc.cpp
    ld.cpp
)


target_link_libraries(vision
    PRIVATE core ${OpenCV_LIBS}
)
