#!/bin/bash
# Build script for tilepack C++ library

set -e  # Exit on error

echo "=== Building Tilepack C++ Library ==="

# Check if OpenCV is installed
if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    echo "ERROR: OpenCV not found!"
    echo "Please install OpenCV:"
    echo "  Ubuntu/Debian: sudo apt-get install libopencv-dev"
    echo "  Fedora: sudo dnf install opencv-devel"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Running CMake..."
cmake ..

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executable: build/tilepack_example"
echo ""
echo "Usage:"
echo "  ./build/tilepack_example encode   # Encode test_image.jpg"
echo "  ./build/tilepack_example decode   # Decode image_radio_file.bin"
