#ifndef IMAGE_SENDER_HPP
#define IMAGE_SENDER_HPP

#include <string>
#include <vector>
#include "communication/uart.hpp"

// #include <termios.h> // POSIX terminal control definitions
#include <unistd.h> 
#include <array>

#define CMD_HANDSHAKE_REQUEST   0x01
#define CMD_DATA_CHUNK          0x02
#define CMD_ACK_READY           0x10
#define CMD_ACK_OK              0x11
#define CMD_NACK_CORRUPT        0x20
#define CMD_NACK_LOST           0x21
#define CMD_IMAGE_REQUEST       0x06

#define MAX_PAYLOAD_SIZE  239  // Max data size per packet: Packetsize(250) - chunk_ID(4) - data_length(4) - last_packet(1) - crc5(1) - packet_id(1)
#define PACKET_SIZE     250
#define CRC5_SIZE       1


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