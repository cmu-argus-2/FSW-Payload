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
  git-lfs \
  nano \
  v4l-utils \
  clang \
  libgoogle-glog-dev \
  libgflags-dev \
  libatlas-base-dev \
  libeigen3-dev \
  libsuitesparse-dev \
  libreadline-dev \
  libnvinfer-dev \
  nlohmann-json3-dev \
  libceres-dev \
  libprotobuf-dev \
  protobuf-compiler \
  libhdf5-dev \
  libtbb-dev \
  ccache
#  libopencv-dev \

# Uncomment this line if you get ccache related issues
# sudo apt-get install --reinstall -y ccache

# Initialise all submodules (source deps + Spice + models)
git submodule update --init --recursive

# Install spdlog from source (pinned via submodule)
cd deps/spdlog
mkdir -p build && cd build
cmake ..
make -j
sudo make install
cd ../../..

# Install HighFive from source (pinned via submodule; HighFive has its own submodules)
cmake -DHIGHFIVE_EXAMPLES=Off \
      -DHIGHFIVE_USE_BOOST=Off \
      -DHIGHFIVE_UNIT_TESTS=Off \
      -DCMAKE_INSTALL_PREFIX=${HIGHFIVE_INSTALL_PREFIX} \
      -B deps/HighFive-src/build \
      deps/HighFive-src
cmake --build deps/HighFive-src/build
sudo cmake --install deps/HighFive-src/build

# Install Ceres Solver from source (pinned via submodule)
# Takes a long time to build; skip if already installed at the right version.
if pkg-config --exact-version=2.2.0 ceres 2>/dev/null; then
    echo "Ceres 2.2.0 already installed — skipping build."
else
    cd deps/ceres-solver
    mkdir -p build && cd build
    cmake ..
    make -j3
    sudo make install
    cd ../../..
fi

# googletest is integrated via CMake FetchContent (deps/googletest submodule).
# No separate build/install step needed.

# Detect CUDA path
CUDA_PATH=$(dirname $(dirname $(which nvcc)))

export CUDA_HOME=$CUDA_PATH
export CUDART_LIBRARY=$CUDA_HOME/lib64/libcudart.so
export LD_LIBRARY_PATH=$CUDA_HOME/lib64: # $LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include: # $CPLUS_INCLUDE_PATH

# Check cudart exists
if [ ! -f "$CUDART_LIBRARY" ]; then
  echo "Error: libcudart.so not found in $CUDA_HOME/lib64"
  exit 1
fi

# Install Spice toolkit (for time and coordinate transformations)
echo "Downloading physics model data files (may take a few minutes) ..."

mkdir -p data/kernels
curl -O -C - --silent https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp --output-dir data/kernels
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc --output-dir data/kernels
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc --output-dir data/kernels
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls --output-dir data/kernels