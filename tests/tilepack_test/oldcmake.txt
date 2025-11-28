# CMakeLists.txt for tilepack library

cmake_minimum_required(VERSION 3.10)
project(tilepack)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (required)
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV development files.")
endif()

# Tilepack library
add_library(tilepack_lib
    tilepack.cpp
    tilepack.hpp
)

target_include_directories(tilepack_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tilepack_lib
    ${OpenCV_LIBS}
)

# Example executable
add_executable(tilepack_example
    tilepack_example.cpp
)

target_link_libraries(tilepack_example
    tilepack_lib
)

# Installation
install(TARGETS tilepack_lib tilepack_example
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES tilepack.hpp
    DESTINATION include
)
