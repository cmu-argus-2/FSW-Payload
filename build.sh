#!/bin/bash
set -e

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

# Ensure submodules are initialised and try to refresh model artifacts.
git submodule sync --recursive
git submodule update --init --recursive

if [ -d "models/.dvc" ]; then
  DVC_BIN=$(command -v dvc || true)
  if [ -z "$DVC_BIN" ] && [ -x "$HOME/.local/bin/dvc" ]; then
    DVC_BIN="$HOME/.local/bin/dvc"
  fi

  if [ -n "$DVC_BIN" ] && [ -x "$DVC_BIN" ]; then
    echo "Pulling DVC-managed model artifacts from the models submodule..."
    cd models
    if "$DVC_BIN" pull; then
      echo "Model artifacts pulled successfully."
    else
      echo "Error: Failed to pull model artifacts with DVC." >&2
    fi
    cd ..
  elif [ -n "$DVC_BIN" ]; then
    echo "Warning: DVC was not installed correctly at $DVC_BIN. Continuing build without refreshing model artifacts." >&2
  else
    echo "Warning: DVC is not installed. Run ./install_deps.sh to enable automatic model artifact pulls." >&2
  fi
fi

# Create binary directory
mkdir -p bin

# Create log directory
mkdir -p logs

# Create build directory and build
mkdir -p build
cd build/

# Run CMake with the specified build type
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DBUILD_TESTS=ON \
      -DNN_ENABLED=${ENABLE_VISION_NN} \
      -DCUDA_ENABLED=${ENABLE_VISION_NN} \
      ..

# Build the project with multiple cores
make -j$(nproc)
