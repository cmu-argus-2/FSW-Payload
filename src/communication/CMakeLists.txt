

add_library(communication
    named_pipe.cpp
    uart.cpp
)

# Tilepack library (implementation in this directory, header in include/communication/)
add_library(tilepack
    tilepack.cpp
)

target_include_directories(tilepack PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tilepack
    ${OpenCV_LIBS}
)
