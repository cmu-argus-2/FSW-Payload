

add_library(vision
    frame.cpp
    camera.cpp
    camera_manager.cpp
    dataset.cpp
    dataset_manager.cpp
    ld.cpp
    prefiltering.cpp
)


target_link_libraries(vision
    PRIVATE core ${OpenCV_LIBS} nlohmann_json::nlohmann_json
)
