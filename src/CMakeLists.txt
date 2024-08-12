
add_library(PAYLOAD_LIB SHARED
    payload.cpp
    task.cpp
    queues.cpp
    commands.cpp
    camera.cpp
    frame.cpp
    messages.cpp
)

# Link libraries to PAYLOAD_LIB
target_link_libraries(PAYLOAD_LIB
    PRIVATE ${OpenCV_LIBS}
)