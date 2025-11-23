/**
 * Example usage of tilepack encoder
 * 
 * This demonstrates how to use the C++ tilepack library to:
 * 1. Encode an image into packets
 * 2. Write packets to binary file
 * 3. Write metadata file
 * 
 * For decoding, use the Python reconstruct.py script.
 */

#include "tilepack.hpp"
#include <iostream>

void example_encode(const std::string& input_image) {
    using namespace tilepack;
    
    std::cout << "=== TILEPACK ENCODER ===" << std::endl;
    std::cout << "Input image: " << input_image << std::endl;
    
    TilepackEncoder encoder(
        1,      // page_id
        640,    // target_width
        480,    // target_height
        64,     // tile_w
        32,     // tile_h
        50      // jpeg_quality 
    );
    
    // Load and process image
    if (!encoder.load_image(input_image)) {
        std::cerr << "Failed to load and process image" << std::endl;
        return;
    }
    
    std::cout << "\nEncoding statistics:" << std::endl;
    std::cout << "  Total packets: " << encoder.get_total_packets() << std::endl;
    std::cout << "  Total bytes: " << encoder.get_total_bytes() << " bytes ("
              << encoder.get_total_bytes() / 1024.0 << " kB)" << std::endl;
    std::cout << "  Compressed size: " << encoder.get_compressed_size() << " bytes ("
              << encoder.get_compressed_size() / 1024.0 << " kB)" << std::endl;
    std::cout << "  Avg packet size: " << encoder.get_avg_packet_size() << " bytes" << std::endl;
    
    // Get metadata
    auto meta = encoder.get_metadata();
    std::cout << "\nImage metadata:" << std::endl;
    std::cout << "  Grid: " << meta.tiles_x << "x" << meta.tiles_y << " tiles" << std::endl;
    std::cout << "  Tile size: " << meta.tile_w << "x" << meta.tile_h << " pixels" << std::endl;
    std::cout << "  Target: " << meta.target_width << "x" << meta.target_height << " pixels" << std::endl;
    std::cout << "  JPEG quality: " << static_cast<int>(meta.jpeg_quality) << std::endl;
    
    // Write radio file
    std::string output_dir = "tilepack";
    std::string radio_file = output_dir + "/image_radio_file.bin";
    std::string meta_file = output_dir + "/image_meta.bin";
    
    if (encoder.write_radio_file(radio_file)) {
        std::cout << "\n✓ Saved radio file: " << radio_file << std::endl;
    } else {
        std::cerr << "✗ Failed to write radio file" << std::endl;
    }
    
    // Write metadata (optional but recommended)
    if (encoder.write_metadata(meta_file)) {
        std::cout << "✓ Saved metadata: " << meta_file << std::endl;
    } else {
        std::cerr << "✗ Failed to write metadata" << std::endl;
    }
    
    std::cout << "\nTo reconstruct the image, use Python:" << std::endl;
    std::cout << "  python3 reconstruct.py" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input_image.jpg>" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " test_image.jpg" << std::endl;
        return 1;
    }
    
    std::string input_image = argv[1];
    example_encode(input_image);
    
    return 0;
}
