

add_subdirectory(core)
add_subdirectory(telemetry)
add_subdirectory(navigation)
add_subdirectory(vision)
add_subdirectory(communication)
# add_subdirectory(inference)

add_library(PAYLOAD_LIB SHARED
    commands.cpp
    messages.cpp
    payload.cpp
    queues.cpp
    configuration.cpp
    image_sender.cpp
)

# Link libraries to PAYLOAD_LIB
target_link_libraries(PAYLOAD_LIB
    PUBLIC core navigation vision communication telemetry tilepack # inference 
)
