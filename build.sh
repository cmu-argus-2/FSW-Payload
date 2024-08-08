#!/bin/bash

# Default build type is Debug
BUILD_TYPE="Debug"

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
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..

# Build the project
make