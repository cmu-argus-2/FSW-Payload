#!/bin/bash
set -e

echo "Welcome to the FSW-Payload installation script!"
echo "This script will install all necessary dependencies and tools for the payload."
echo "Please ensure you have sudo privileges before proceeding."

# Check for apt
if ! command -v apt &> /dev/null; then
  echo "This script supports only Debian-based systems." >&2
  exit 1
fi

sudo apt update

# Install required system packages
sudo apt install -y \
  build-essential \
  cmake \
  git \
  nano \
  v4l-utils \
  clang \
  libopencv-dev \
  libeigen3-dev \
  libreadline-dev \
  libnvinfer-dev \
  ccache

# Uncomment this line if you get ccache related issues
# sudo apt-get install --reinstall -y ccache

# Create deps folder
mkdir -p deps && cd deps

# Install GTest from source
rm -rf googletest # Remove existing GTest folder
git clone https://github.com/google/googletest.git
cd googletest
mkdir -p build && cd build
cmake ..
make -j
sudo make install
cd ../..

# Install spdlog from source
rm -rf spdlog # Remove existing spdlog folder
git clone --branch v1.12.0 https://github.com/gabime/spdlog.git
cd spdlog
mkdir -p build && cd build
cmake ..
make -j
sudo make install
cd ../..

cd ..


# Detect CUDA path
CUDA_PATH=$(dirname $(dirname $(which nvcc)))
export CUDA_HOME=$CUDA_PATH
export CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH


# Check cudart exists
if [ ! -f "$CUDART_LIBRARY" ]; then
  echo "Error: libcudart.so not found in $CUDA_HOME/lib64"
  exit 1
fi


# TODO: IMX708 sensor 


