#!/bin/bash
# Build script for tilepack C++ library

set -e  

echo "=== Building Tilepack C++ Library ==="

if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    echo "ERROR: OpenCV not found!"
    echo "Please install OpenCV:"
    echo "  Ubuntu/Debian: sudo apt-get install libopencv-dev"
    echo "  Fedora: sudo dnf install opencv-devel"
    exit 1
fi

mkdir -p build
cd build

echo "Running CMake..."
cmake ..

echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executable: build/tilepack_example"
echo ""
echo "Usage:"
echo "  ./build/tilepack_example <input_image.jpg>   # Encode test_image.jpg"
