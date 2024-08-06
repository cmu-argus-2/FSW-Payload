cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 20)

project(
    PAYLOAD
    VERSION 1.0
    LANGUAGES CXX
)


# Find the required packages
find_package(spdlog REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

# Include the directories for header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the subdirectory containing source files
add_subdirectory(src)

# Add the executable
add_executable(${PROJECT_NAME} main.cpp) 

# Link libraries to the executable
target_link_libraries(${PROJECT_NAME} 
    PRIVATE spdlog::spdlog 
    PAYLOAD_LIB
)
