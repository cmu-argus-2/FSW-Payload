cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 17)

project(PAYLOAD VERSION 1.0 LANGUAGES CXX)


# Check if LIBTORCH_PATH is set in the environment
if(NOT DEFINED ENV{LIBTORCH_PATH})
    message(FATAL_ERROR "LIBTORCH_PATH environment variable is not set. Please set it in your .bashrc (e.g., export LIBTORCH_PATH=/path/to/libtorch/share/cmake/Torch/).")
else()
    set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_PATH})
endif()


# Find the required packages
set(SPDLOG_FMT_EXTERNAL OFF)
find_package(spdlog REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Include OpenCV directories
include_directories(${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# Include the directories for header files
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the subdirectory containing source files
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(src)

# Add the executable
add_executable(${PROJECT_NAME} main.cpp)

# Link libraries to the executable
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    spdlog::spdlog
    PAYLOAD_LIB
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)
