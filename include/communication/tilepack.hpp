#ifndef TILEPACK_HPP
#define TILEPACK_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <map>

namespace cv {
    class Mat;
}

namespace tilepack {

constexpr uint16_t DEFAULT_TARGET_WIDTH = 640;
constexpr uint16_t DEFAULT_TARGET_HEIGHT = 480;
constexpr uint16_t DEFAULT_TILE_W = 64;
constexpr uint16_t DEFAULT_TILE_H = 32;
constexpr int DEFAULT_JPEG_QUALITY = 30;
constexpr size_t MAX_PACKET_SIZE = 240;
constexpr size_t PACKET_HEADER_SIZE = 7;
constexpr size_t MAX_PAYLOAD_PER_PACKET = MAX_PACKET_SIZE - PACKET_HEADER_SIZE; // 233 bytes

struct PacketHeader {
    uint16_t payload_size_bytes;  
    uint16_t page_id;          
    uint16_t tile_idx;            
    uint8_t frag_idx;             

    std::vector<uint8_t> to_bytes() const;
    
    static PacketHeader from_bytes(const uint8_t* data, size_t len);
    
    static constexpr size_t size() { return PACKET_HEADER_SIZE; }
};

struct Packet {
    PacketHeader header;
    std::vector<uint8_t> payload;
    
    size_t total_size() const { return PACKET_HEADER_SIZE + payload.size(); }
};

struct ImageMetadata {
    uint16_t page_id;
    uint16_t tiles_x;
    uint16_t tiles_y;
    uint16_t tile_w;
    uint16_t tile_h;
    uint16_t target_width;
    uint16_t target_height;
    uint8_t jpeg_quality;
    
    std::vector<uint8_t> to_bytes() const;
    
    static ImageMetadata from_bytes(const uint8_t* data, size_t len);
    
    static constexpr size_t size() { return 15; }
};

struct Tile {
    std::vector<uint8_t> jpeg_data;  
    uint16_t tile_idx;
};

class TilepackEncoder {
public:
    TilepackEncoder(uint16_t page_id = 1,
                   uint16_t target_width = DEFAULT_TARGET_WIDTH,
                   uint16_t target_height = DEFAULT_TARGET_HEIGHT,
                   uint16_t tile_w = DEFAULT_TILE_W,
                   uint16_t tile_h = DEFAULT_TILE_H,
                   int jpeg_quality = DEFAULT_JPEG_QUALITY);
    
    ~TilepackEncoder();
    
    bool load_image(const std::string& image_path);
    
    const std::vector<Packet>& get_packets() const { return packets_; }
    
    const ImageMetadata& get_metadata() const { return metadata_; }
    
    bool write_radio_file(const std::string& output_path) const;
    
    bool write_metadata(const std::string& output_path) const;
    
    size_t get_total_packets() const { return packets_.size(); }
    size_t get_total_bytes() const;
    size_t get_compressed_size() const;
    double get_avg_packet_size() const;
    
    const std::vector<Tile>& get_tiles() const { return tiles_; }

private:
    uint16_t page_id_;
    uint16_t target_width_;
    uint16_t target_height_;
    uint16_t tile_w_;
    uint16_t tile_h_;
    int jpeg_quality_;
    
    ImageMetadata metadata_;
    std::vector<Tile> tiles_;
    std::vector<Packet> packets_;
    
    bool tile_image(const cv::Mat& image);
    bool compress_tile_jpeg(const cv::Mat& tile, std::vector<uint8_t>& jpeg_out);
    void packetize_tiles();
    std::vector<Packet> packetize_tile(uint16_t tile_idx, 
                                       const std::vector<uint8_t>& tile_bytes);
};

} 

#endif
