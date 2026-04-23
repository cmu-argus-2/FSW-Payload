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

# Ensure submodules are initialised and model artifacts are present.
git submodule sync --recursive
git submodule update --init --recursive

if [ -d "models/.dvc" ]; then
  DVC_BIN=$(command -v dvc || true)
  if [ -z "$DVC_BIN" ] && [ -x "$HOME/.local/bin/dvc" ]; then
    DVC_BIN="$HOME/.local/bin/dvc"
  fi

  if [ -z "$DVC_BIN" ]; then
    if ! python3 -m pipx --help > /dev/null 2>&1; then
      echo "Error: pipx is required to install DVC. Run ./install_deps.sh first." >&2
      exit 1
    fi

    python3 -m pipx ensurepath > /dev/null || true
    python3 -m pipx install --force "dvc[ssh]"
    DVC_BIN="$HOME/.local/bin/dvc"
  fi

  if [ ! -x "$DVC_BIN" ]; then
    echo "Error: DVC was not installed correctly." >&2
    exit 1
  fi

  echo "Pulling DVC-managed model artifacts from the models submodule..."
  (cd models && "$DVC_BIN" pull)
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
