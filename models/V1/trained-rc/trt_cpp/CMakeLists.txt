cmake_minimum_required(VERSION 3.11)
project(RunTRT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Set TensorRT paths manually
set(TENSORRT_INCLUDE_DIR "/usr/include")
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")

include_directories(${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
link_directories(${TENSORRT_LIB_DIR})

# Source files
add_executable(run_trt run_trt.cpp)

# Link only the required TensorRT libraries
target_link_libraries(run_trt ${OpenCV_LIBS} ${CUDA_LIBRARIES} -lnvinfer -lcudart)
