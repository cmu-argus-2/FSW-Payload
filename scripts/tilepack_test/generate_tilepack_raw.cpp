#include "communication/tilepack.hpp"
#include <iostream>
#include <filesystem>

int main(int argc, char** argv) {
    using namespace tilepack;

    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input_image> <output_radio_raw.bin>" << std::endl;
        std::cout << "Example: " << argv[0] << " tests/tilepack_test/test_image.jpg tests/tilepack_test/image_radio_file_cpp_raw.bin" << std::endl;
        return 1;
    }

    const std::string input_image = argv[1];
    const std::string output_radio = argv[2];

    if (!std::filesystem::exists(input_image)) {
        std::cerr << "Input image does not exist: " << input_image << std::endl;
        return 1;
    }

    TilepackEncoder encoder;
    if (!encoder.load_image(input_image)) {
        std::cerr << "Failed to load/process image: " << input_image << std::endl;
        return 1;
    }

    if (!encoder.write_radio_file_raw(output_radio)) {
        std::cerr << "Failed to write raw radio file: " << output_radio << std::endl;
        return 1;
    }

    std::cout << "Generated raw radio file (no DH padding): " << output_radio << std::endl;
    std::cout << "Packets: " << encoder.get_total_packets() << std::endl;
    std::cout << "Bytes (header+payload only): " << encoder.get_total_bytes() << std::endl;

    return 0;
}
