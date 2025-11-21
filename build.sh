#!/bin/bash

# Default build type is Debug
BUILD_TYPE="Debug"
ENABLE_VISION_NN=1
#LIBTORCH_PATH="~/libtorch"

# Parse args
for arg in "$@"; do
  if [[ $arg == "Debug" || $arg == "Release" ]]; then
    BUILD_TYPE=$arg
  elif [[ $arg == "disable-nn" ]]; then
    ENABLE_VISION_NN=0
  else
    echo "Unknown option: $arg"
    echo "Usage: ./build.sh [Debug|Release]" # [disable-nn]"
    exit 1
  fi
done

echo "Building in ${BUILD_TYPE} mode"
echo "NN module is ${ENABLE_VISION_NN}"

# Create binary directory
mkdir -p bin

# Create log directory
mkdir -p logs

# Create build directory and build
mkdir -p build
cd build/

# Run CMake with the specified build type
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILD_TESTS=ON -DNN_ENABLED=${ENABLE_VISION_NN} .. 

# Build the project with multiple cores
make -j$(nproc)
