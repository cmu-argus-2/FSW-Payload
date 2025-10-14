#ifndef IMAGE_SENDER_HPP
#define IMAGE_SENDER_HPP

#include <string>
#include <vector>
#include "communication/uart.hpp"

// #include <termios.h> // POSIX terminal control definitions
#include <unistd.h> 
#include <array>

// --- Protocol Definitions ---
// The maximum payload size for a data chunk (image data).
constexpr size_t MAX_PAYLOAD_SIZE = 246;

// Header (ID + Length) + CRC (1)
constexpr size_t PACKET_OVERHEAD = 4 + 4 + 1; 

// Total maximum size of any packet (255 bytes)
constexpr size_t TOTAL_PACKET_SIZE = MAX_PAYLOAD_SIZE + PACKET_OVERHEAD;



// Data struct for image packets
struct ImagePacket
{
    uint32_t id = 0;               // Packet sequence number (0 for handshake)
    uint32_t length = 0;           // Size of data in the current chunk (up to MAX_PAYLOAD_SIZE)
    uint8_t data[MAX_PAYLOAD_SIZE]; // Data buffer
    uint8_t crc = 0;               // CRC-5 of everything before this field

    struct Out {
        std::vector<uint8_t> bytes;
    };
};
enum img_packet_cmd__id : uint8_t {
    CMD_HANDSHAKE_REQUEST = 0x01, 
    CMD_DATA_CHUNK = 0x02, 
    CMD_ACK_READY = 0x10,
    CMD_ACK_OK = 0x11,
    CMD_NACK_CORRUPT = 0x20, 
    CMD_NACK_LOST = 0x21
};

class ImageSender {
public:
    ImageSender();

    bool Initialize();
    void Close();

    Packet::Out CreateImagePacket(uint32_t packet_id, const uint8_t* data, uint32_t data_length);
    bool SendImage(const std::string& image_path);
    void RunImageTransfer();

    
private:
    UART uart;
    bool is_initialized = false;
    bool shake_received = false;
    bool sending_ack_rec = false;

    uint32_t HandshakeWithMainboard();
    uint32_t SendImageOverUart(const std::string& image_path);
};

#endif // IMAGE_SENDER_HPP