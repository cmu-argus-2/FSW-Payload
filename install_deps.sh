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

# Install HighFive from source
rm -rf HighFive-src # Remove existing HighFive folder
git clone --recursive https://github.com/BlueBrain/HighFive.git HighFive-src
cmake -DHIGHFIVE_EXAMPLES=Off \
      -DHIGHFIVE_USE_BOOST=Off \
      -DHIGHFIVE_UNIT_TESTS=Off \
      -DCMAKE_INSTALL_PREFIX=${HIGHFIVE_INSTALL_PREFIX} \
      -B HighFive-src/build \
      HighFive-src

cmake --build HighFive-src/build
sudo cmake --install HighFive-src/build

# Install Ceres Solver from source
# Takes a lot of time to build, best not to rebuild if already installed
if [ ! -d "ceres-solver/.git" ]; then
    echo "Ceres not found. Cloning..."
    git clone https://github.com/ceres-solver/ceres-solver.git "ceres-solver"
else
    echo "Ceres already cloned — skipping."
fi
cd ceres-solver
git checkout --detach 2.2.0
mkdir -p build
cd build
cmake ..
make -j3
# make test
sudo make install
cd ../..

cd ..

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
mkdir -p deps/data
curl -O -C - --silent https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp --output-dir deps/data
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc --output-dir deps/data
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc --output-dir deps/data
curl -O -C - --silent https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls --output-dir deps/data

git submodule update --init --recursive