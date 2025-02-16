cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 17)

project(PAYLOAD VERSION 1.0 LANGUAGES CXX)
add_definitions(-DROOT_DIR=\"${CMAKE_SOURCE_DIR}\")
add_definitions(-DMODELS_DIR=\"${CMAKE_SOURCE_DIR}/models\")

set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

# Don't use
# set(CMAKE_CXX_COMPILER clang++)

# Check if LIBTORCH_PATH is set in the environment
if(NOT DEFINED ENV{LIBTORCH_PATH})
    message(FATAL_ERROR "LIBTORCH_PATH environment variable is not set. Please set it in your .bashrc (e.g., export LIBTORCH_PATH=/path/to/libtorch/share/cmake/Torch/).")
else()
    set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_PATH})
endif()

# Set a global output directory for all executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)
# Set a compile definition for the executable directory
add_compile_definitions(EXECUTABLE_DIR=\"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}\")

# Preprocessor definition for the path of the inter-process fifo (command-line interface)
add_compile_definitions(IPC_FIFO_PATH="/tmp/payload_fifo")

##### FIND THE REQUIRED PACKAGES

set(SPDLOG_FMT_EXTERNAL OFF)
find_package(spdlog REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)


# Include OpenCV directories
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
# Include the directories for header files
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external) # ignore warnings from external libraries
# Add the source files subdirectory
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(src)



##### MAIN FLIGHT SOFTWARE PROCESS

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    spdlog::spdlog
    PAYLOAD_LIB
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)




##### COMMAND-LINE INTERFACE

find_library(READLINE_LIBRARY readline)
add_executable(CLI_CMD src/cli_cmd.cpp)
if (READLINE_LIBRARY)
    target_link_libraries(CLI_CMD PRIVATE ${READLINE_LIBRARY})
else()
    message(FATAL_ERROR "Readline library not found. Please install libreadline-dev.")
endif()


##### TESTS

option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    enable_testing()    
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()


###### GLOBAL COMPILE OPTIONS 

add_compile_options(
    -Werror
    -Wall
    -Wextra
    -Wconversion
    -Wsign-conversion
)