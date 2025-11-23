#include "tilepack.hpp"
#include <fstream>
#include <algorithm>
#include <map>
#include <cstring>
#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace tilepack {

// ============================================================
// PacketHeader Implementation
// ============================================================

std::vector<uint8_t> PacketHeader::to_bytes() const {
    std::vector<uint8_t> bytes(PACKET_HEADER_SIZE);
    
    bytes[0] = (payload_size_bytes >> 8) & 0xFF;
    bytes[1] = payload_size_bytes & 0xFF;
    bytes[2] = (page_id >> 8) & 0xFF;
    bytes[3] = page_id & 0xFF;
    bytes[4] = (tile_idx >> 8) & 0xFF;
    bytes[5] = tile_idx & 0xFF;
    bytes[6] = frag_idx;
    
    return bytes;
}

PacketHeader PacketHeader::from_bytes(const uint8_t* data, size_t len) {
    if (len < PACKET_HEADER_SIZE) {
        throw std::runtime_error("Insufficient data for packet header");
    }
    
    PacketHeader header;
    header.payload_size_bytes = (static_cast<uint16_t>(data[0]) << 8) | data[1];
    header.page_id = (static_cast<uint16_t>(data[2]) << 8) | data[3];
    header.tile_idx = (static_cast<uint16_t>(data[4]) << 8) | data[5];
    header.frag_idx = data[6];
    
    return header;
}

// ============================================================
// ImageMetadata Implementation
// ============================================================

std::vector<uint8_t> ImageMetadata::to_bytes() const {
    std::vector<uint8_t> bytes(15);
    
    // Big-endian encoding: >HHHHHHHB
    bytes[0] = (page_id >> 8) & 0xFF;
    bytes[1] = page_id & 0xFF;
    bytes[2] = (tiles_x >> 8) & 0xFF;
    bytes[3] = tiles_x & 0xFF;
    bytes[4] = (tiles_y >> 8) & 0xFF;
    bytes[5] = tiles_y & 0xFF;
    bytes[6] = (tile_w >> 8) & 0xFF;
    bytes[7] = tile_w & 0xFF;
    bytes[8] = (tile_h >> 8) & 0xFF;
    bytes[9] = tile_h & 0xFF;
    bytes[10] = (target_width >> 8) & 0xFF;
    bytes[11] = target_width & 0xFF;
    bytes[12] = (target_height >> 8) & 0xFF;
    bytes[13] = target_height & 0xFF;
    bytes[14] = jpeg_quality;
    
    return bytes;
}

ImageMetadata ImageMetadata::from_bytes(const uint8_t* data, size_t len) {
    if (len < 15) {
        throw std::runtime_error("Insufficient data for metadata");
    }
    
    ImageMetadata meta;
    meta.page_id = (static_cast<uint16_t>(data[0]) << 8) | data[1];
    meta.tiles_x = (static_cast<uint16_t>(data[2]) << 8) | data[3];
    meta.tiles_y = (static_cast<uint16_t>(data[4]) << 8) | data[5];
    meta.tile_w = (static_cast<uint16_t>(data[6]) << 8) | data[7];
    meta.tile_h = (static_cast<uint16_t>(data[8]) << 8) | data[9];
    meta.target_width = (static_cast<uint16_t>(data[10]) << 8) | data[11];
    meta.target_height = (static_cast<uint16_t>(data[12]) << 8) | data[13];
    meta.jpeg_quality = data[14];
    
    return meta;
}

// ============================================================
// TilepackEncoder Implementation
// ============================================================

TilepackEncoder::TilepackEncoder(uint16_t page_id,
                                 uint16_t target_width,
                                 uint16_t target_height,
                                 uint16_t tile_w,
                                 uint16_t tile_h,
                                 int jpeg_quality)
    : page_id_(page_id),
      target_width_(target_width),
      target_height_(target_height),
      tile_w_(tile_w),
      tile_h_(tile_h),
      jpeg_quality_(jpeg_quality) {
    
    // Initialize metadata
    metadata_.page_id = page_id;
    metadata_.tile_w = tile_w;
    metadata_.tile_h = tile_h;
    metadata_.target_width = target_width;
    metadata_.target_height = target_height;
    metadata_.jpeg_quality = static_cast<uint8_t>(std::max(0, std::min(255, jpeg_quality)));
}

TilepackEncoder::~TilepackEncoder() {
}

bool TilepackEncoder::load_image(const std::string& image_path) {
    // Load image using OpenCV
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return false;
    }
    
    std::cout << "Loaded image: " << img.cols << "x" << img.rows << std::endl;
    
    // Resize to target dimensions
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_width_, target_height_), 0, 0, cv::INTER_LINEAR);
    
    std::cout << "Resized to: " << resized.cols << "x" << resized.rows << std::endl;
    
    // Tile the image
    if (!tile_image(resized)) {
        return false;
    }
    
    // Packetize all tiles
    packetize_tiles();
    
    std::cout << "Generated " << tiles_.size() << " tiles" << std::endl;
    std::cout << "Generated " << packets_.size() << " packets" << std::endl;
    
    return true;
}

bool TilepackEncoder::tile_image(const cv::Mat& image) {
    tiles_.clear();
    
    int width = image.cols;
    int height = image.rows;
    
    // Calculate number of tiles
    int tiles_x = static_cast<int>(std::ceil(static_cast<double>(width) / tile_w_));
    int tiles_y = static_cast<int>(std::ceil(static_cast<double>(height) / tile_h_));
    
    metadata_.tiles_x = static_cast<uint16_t>(tiles_x);
    metadata_.tiles_y = static_cast<uint16_t>(tiles_y);
    
    // Pad image if necessary
    cv::Mat padded_img;
    int padded_width = tiles_x * tile_w_;
    int padded_height = tiles_y * tile_h_;
    
    if (width != padded_width || height != padded_height) {
        padded_img = cv::Mat::zeros(padded_height, padded_width, image.type());
        image.copyTo(padded_img(cv::Rect(0, 0, width, height)));
    } else {
        padded_img = image.clone();
    }
    
    // Extract tiles
    uint16_t tile_idx = 0;
    for (int ty = 0; ty < tiles_y; ty++) {
        for (int tx = 0; tx < tiles_x; tx++) {
            // Extract tile region
            cv::Rect roi(tx * tile_w_, ty * tile_h_, tile_w_, tile_h_);
            cv::Mat tile = padded_img(roi);
            
            // Compress tile as JPEG
            std::vector<uint8_t> jpeg_data;
            if (!compress_tile_jpeg(tile, jpeg_data)) {
                std::cerr << "Failed to compress tile " << tile_idx << std::endl;
                return false;
            }
            
            Tile t;
            t.tile_idx = tile_idx;
            t.jpeg_data = jpeg_data;
            tiles_.push_back(t);
            
            tile_idx++;
        }
    }
    
    return true;
}

bool TilepackEncoder::compress_tile_jpeg(const cv::Mat& tile, std::vector<uint8_t>& jpeg_out) {
    // JPEG compression parameters
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(jpeg_quality_);
    params.push_back(cv::IMWRITE_JPEG_OPTIMIZE);
    params.push_back(1);
    
    // Encode as JPEG
    if (!cv::imencode(".jpg", tile, jpeg_out, params)) {
        std::cerr << "Failed to encode tile as JPEG" << std::endl;
        return false;
    }
    
    return true;
}

void TilepackEncoder::packetize_tiles() {
    packets_.clear();
    
    for (const auto& tile : tiles_) {
        auto tile_packets = packetize_tile(tile.tile_idx, tile.jpeg_data);
        packets_.insert(packets_.end(), tile_packets.begin(), tile_packets.end());
    }
}

std::vector<Packet> TilepackEncoder::packetize_tile(uint16_t tile_idx,
                                                     const std::vector<uint8_t>& tile_bytes) {
    std::vector<Packet> packets;
    
    // Split tile_bytes into fragments that fit within MAX_PAYLOAD_PER_PACKET
    size_t offset = 0;
    uint8_t frag_idx = 0;
    
    while (offset < tile_bytes.size()) {
        size_t remaining = tile_bytes.size() - offset;
        size_t payload_size = std::min(remaining, MAX_PAYLOAD_PER_PACKET);
        
        Packet pkt;
        pkt.header.payload_size_bytes = static_cast<uint16_t>(payload_size);
        pkt.header.page_id = page_id_;
        pkt.header.tile_idx = tile_idx;
        pkt.header.frag_idx = frag_idx;
        
        pkt.payload.assign(tile_bytes.begin() + offset,
                          tile_bytes.begin() + offset + payload_size);
        
        packets.push_back(pkt);
        
        offset += payload_size;
        frag_idx++;
    }
    
    return packets;
}

bool TilepackEncoder::write_radio_file(const std::string& output_path) const {
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return false;
    }
    
    for (const auto& pkt : packets_) {
        // Write header
        auto header_bytes = pkt.header.to_bytes();
        file.write(reinterpret_cast<const char*>(header_bytes.data()), header_bytes.size());
        
        // Write payload
        file.write(reinterpret_cast<const char*>(pkt.payload.data()), pkt.payload.size());
        
        // Verify packet size
        if (pkt.total_size() > MAX_PACKET_SIZE) {
            std::cerr << "Warning: Packet exceeds MAX_PACKET_SIZE" << std::endl;
        }
    }
    
    file.close();
    return true;
}

bool TilepackEncoder::write_metadata(const std::string& output_path) const {
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open metadata file: " << output_path << std::endl;
        return false;
    }
    
    auto meta_bytes = metadata_.to_bytes();
    file.write(reinterpret_cast<const char*>(meta_bytes.data()), meta_bytes.size());
    
    file.close();
    return true;
}

size_t TilepackEncoder::get_total_bytes() const {
    size_t total = 0;
    for (const auto& pkt : packets_) {
        total += pkt.total_size();
    }
    return total;
}

size_t TilepackEncoder::get_compressed_size() const {
    size_t total = 0;
    for (const auto& tile : tiles_) {
        total += tile.jpeg_data.size();
    }
    return total;
}

double TilepackEncoder::get_avg_packet_size() const {
    if (packets_.empty()) return 0.0;
    return static_cast<double>(get_total_bytes()) / packets_.size();
}

} // namespace tilepack
