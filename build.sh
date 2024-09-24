#!/bin/bash

# Default build type is Debug
BUILD_TYPE="Debug"
#LIBTORCH_PATH="~/libtorch"

# Check for command-line argument
if [ $# -eq 1 ]; then
  if [ "$1" == "Debug" ] || [ "$1" == "Release" ]; then
    BUILD_TYPE=$1
  else
    echo "Invalid build type. Use 'Debug' or 'Release'."
    exit 1
  fi
fi

echo "Building in ${BUILD_TYPE} mode"

# Create build directory and build
mkdir -p build
cd build/

# Run CMake with the specified build type
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILD_TESTS=ON .. #-DCMAKE_CXX_FLAGS="-Werror -Wall -Wextra -Wconversion -Wsign-conversion"
#cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
#cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_PREFIX_PATH="${LIBTORCH_PATH}/share/cmake/Torch/" ..

# Build the project
make