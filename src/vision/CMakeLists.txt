

add_library(vision
    camera_manager.cpp
    camera.cpp
    frame.cpp
)


target_link_libraries(vision
    PRIVATE core ${OpenCV_LIBS}
)
