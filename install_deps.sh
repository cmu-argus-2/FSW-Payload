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
  python3-pip \
  python3-venv \
  pipx \
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
  libscotch-dev \
  libcli11-dev \
  ninja-build
#  libopencv-dev \

# Uncomment this line if you get ccache related issues
# sudo apt-get install --reinstall -y ccache

# Configure ccache for faster builds
ccache --set-config max_size=15G
ccache --set-config sloppiness=include_file_mtime,include_file_ctime

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

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
INSTALL_PREFIX="$REPO_ROOT/deps/install"
BUILD_NPROC=$(nproc)
mkdir -p "$INSTALL_PREFIX"

# ── Ensure CasADi source is present ──────────────────────────────────────────
# Try submodule first; fall back to a direct clone if not yet registered.
if [ ! -f "$REPO_ROOT/deps/casadi/CMakeLists.txt" ]; then
  echo "CasADi source not found — cloning …"
  git -C "$REPO_ROOT" submodule add https://github.com/casadi/casadi.git deps/casadi 2>/dev/null || true
  git -C "$REPO_ROOT" submodule update --init deps/casadi 2>/dev/null || \
    git clone --depth 1 --branch 3.6.5 https://github.com/casadi/casadi.git \
              "$REPO_ROOT/deps/casadi"
fi

# ── Build IPOPT (deps/Ipopt submodule → deps/install/) ───────────────────────
echo "Building IPOPT …"

MUMPS_INC="$REPO_ROOT/deps/mumps/include"
MUMPS_LFLAGS="\
$REPO_ROOT/deps/mumps/lib/libdmumps_seq.a \
$REPO_ROOT/deps/mumps/lib/libmumps_common_seq.a \
$REPO_ROOT/deps/mumps/lib/libpord_seq.a \
$REPO_ROOT/deps/mumps/lib/libmpiseq_seq.a \
$REPO_ROOT/deps/mumps/lib/libesmumps.a \
$REPO_ROOT/deps/mumps/lib/libscotch.a \
$REPO_ROOT/deps/mumps/lib/libscotcherr.a \
-lgfortran -llapack -lblas"

mkdir -p "$REPO_ROOT/deps/Ipopt/ipopt_build"
cd "$REPO_ROOT/deps/Ipopt/ipopt_build"
"$REPO_ROOT/deps/Ipopt/configure" \
  --prefix="$INSTALL_PREFIX" \
  --disable-java \
  --with-mumps \
  "--with-mumps-cflags=-I${MUMPS_INC} -I${MUMPS_INC}/mumps_seq" \
  "--with-mumps-lflags=${MUMPS_LFLAGS}"
make -j"$BUILD_NPROC"
make install
cd "$REPO_ROOT"

# ── Build CasADi with IPOPT support (deps/casadi → deps/install/) ────────────
echo "Building CasADi with IPOPT …"

mkdir -p "$REPO_ROOT/deps/casadi/casadi_build"
cd "$REPO_ROOT/deps/casadi/casadi_build"
PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_IPOPT=ON \
  -DWITH_PYTHON=OFF \
  -DWITH_MATLAB=OFF \
  -DWITH_OCTAVE=OFF
make -j"$BUILD_NPROC"
make install
cd "$REPO_ROOT"

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