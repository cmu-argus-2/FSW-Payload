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
  libeigen3-dev \
  libreadline-dev \
  libnvinfer-dev \
  nlohmann-json3-dev \
  libprotobuf-dev \
  protobuf-compiler \
  libhdf5-dev \
  libtbb-dev \
  ccache \
  autoconf \
  automake \
  libtool \
  pkg-config \
  gfortran \
  liblapack-dev \
  libblas-dev \
  libmumps-seq-dev \
  libmumps-headers-dev \
  libscotch-dev
#  libopencv-dev \

# Uncomment this line if you get ccache related issues
# sudo apt-get install --reinstall -y ccache

# Initialise all submodules (source deps + Spice + models + Ipopt)
git submodule update --init --recursive

# Stage MUMPS libraries and headers into deps/mumps/ so CMake can find them.
# This directory is gitignored; it is (re)populated here from the system packages
# installed above (libmumps-seq-dev, libmumps-headers-dev, libscotch-dev).
MULTIARCH=$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo "aarch64-linux-gnu")
MUMPS_LIB_SYS="/usr/lib/${MULTIARCH}"

mkdir -p deps/mumps/lib deps/mumps/include/mumps_seq

# Static libraries baked into libipopt.so at build time
cp "${MUMPS_LIB_SYS}/libdmumps_seq.a" \
   "${MUMPS_LIB_SYS}/libmumps_common_seq.a" \
   "${MUMPS_LIB_SYS}/libpord_seq.a" \
   "${MUMPS_LIB_SYS}/libmpiseq_seq.a" \
   "${MUMPS_LIB_SYS}/libesmumps.a" \
   "${MUMPS_LIB_SYS}/libscotch.a" \
   "${MUMPS_LIB_SYS}/libscotcherr.a" \
   deps/mumps/lib/

# Headers (including the sequential MPI stub in mumps_seq/)
cp /usr/include/dmumps_c.h \
   /usr/include/dmumps_root.h \
   /usr/include/dmumps_struc.h \
   /usr/include/mumps_c_types.h \
   /usr/include/mumps_compat.h \
   /usr/include/mumps_int_def.h \
   deps/mumps/include/
cp /usr/include/mumps_seq/mpi.h \
   /usr/include/mumps_seq/mpif.h \
   /usr/include/mumps_seq/elapse.h \
   deps/mumps/include/mumps_seq/

# Ipopt (deps/Ipopt submodule) is built from source automatically by CMake's
# ExternalProject_Add during the main build — no manual build step needed here.

# Install spdlog from source (pinned via submodule)
cd deps/spdlog
mkdir -p build && cd build
cmake ..
make -j
sudo make install
cd ../../..

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