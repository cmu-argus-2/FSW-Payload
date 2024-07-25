cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(
    PAYLOAD
    VERSION 1.0
    LANGUAGES CXX
)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)

add_executable("${PROJECT_NAME}" main.cpp) 

target_link_libraries("${PROJECT_NAME}" PAYLOAD_LIB)
