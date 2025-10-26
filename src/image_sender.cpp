#include <filesystem>
#include "communication/uart.hpp"
#include "image_sender.hpp"
#include "core/data_handling.hpp"
#include "core/timing.hpp"
#include "communication/comms.hpp"
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <unistd.h>   


// #include "uart_protocol.h"
// #include "communication_base.h" // Includes the Communication interface
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>




// --- Checksum Implementation (CRC-5: x^5 + x^2 + 1, Init 0x1F) ---
inline uint8_t calculate_crc5(const uint8_t* data, size_t length)
{
    // Polynomial: 0x05 (0b00101)
    const uint8_t polynomial = 0x05; 
    
    // Initial CRC value: 0x1F (0b11111)
    uint8_t crc = 0x1F;
    
    // Process the input buffer byte-by-byte
    for (size_t i = 0; i < length; ++i) {
        uint8_t byte = data[i];

        // Process 8 bits of the current byte (MSB first)
        for (int j = 0; j < 8; ++j) {
            
            // 1. Determine if the MSB of the CRC (bit 4, value 0x10) is set.
            uint8_t msb_of_crc = crc & 0x10;
            
            // 2. Determine the incoming data bit (MSB first) and position it at bit 4 (0x10).
            uint8_t data_bit_at_msb_pos = (byte >> (7 - j)) & 0x01;
            data_bit_at_msb_pos <<= 4;

            // 3. Shift CRC left by 1 bit, masked to 5 bits.
            crc = (crc << 1) & 0x1F; 

            // 4. Check if the XOR of the MSB of the *original* CRC and the data bit is 1.
            if (msb_of_crc ^ data_bit_at_msb_pos) {
                // If the check bit is 1, XOR the 5-bit CRC register with the polynomial.
                crc ^= polynomial;
            }
        }
    }

    // The result is the final 5-bit CRC value (0x00 to 0x1F)
    return crc;
}

// Serialize image packet for transmission
Packet::Out ImageSender::CreateImagePacket(uint32_t packet_id, const uint8_t* data, uint32_t data_length)
{
    if (data_length == 0 || data_length > MAX_PAYLOAD_SIZE) {
        // std::cerr << "Error: Data chunk size is invalid (0 or > MAX_PAYLOAD_SIZE)." << std::endl;
        return {}; 
    }
    
    // Prepare the in-memory packet structure (for CRC calculation)
    ImagePacket packet;
    packet.id = packet_id;
    packet.length = data_length; 
    std::memcpy(packet.data, data, data_length);

    // CRC is calculated over: ID (1) + Length (1) + Data (N)
    
    std::vector<uint8_t> crc_input;
    crc_input.reserve(sizeof(packet.id) + sizeof(packet.length) + data_length);
    
    crc_input.push_back(packet.id);
    
    crc_input.push_back(packet.length);
    
    crc_input.insert(crc_input.end(), packet.data, packet.data + data_length);

    packet.crc = calculate_crc5(crc_input.data(), crc_input.size());

    // Size = ID (1) + Length (1) + Data (N) + CRC (1)
    size_t final_packet_size = sizeof(packet.id) + sizeof(packet.length) + data_length + sizeof(packet.crc);
    
    // Serialize into the output vector (ImagePacket::Out)
    Packet::Out output {};
    // output.bytes.resize(final_packet_size);
    uint8_t* buffer_ptr = output.data();
    
    // Copy fields to output buffer in order
    // 4 byte for ID
    std::memcpy(buffer_ptr, &packet.id, sizeof(packet.id));
    buffer_ptr += sizeof(packet.id);
    // 4 bytes for length
    std::memcpy(buffer_ptr, &packet.length, sizeof(packet.length));
    buffer_ptr += sizeof(packet.length);
    // N bytes for data
    std::memcpy(buffer_ptr, packet.data, data_length);
    buffer_ptr += data_length;
    //1byte for crc
    std::memcpy(buffer_ptr, &packet.crc, sizeof(packet.crc));

    return output;
}


ImageSender::ImageSender()
    : uart(), is_initialized(false)
{
    // Initialize UART with given port and baud rate
    // uart = UART(uart_port.c_str(), baud_rate); // Uncomment and implement constructor in UART class
}


bool ImageSender::Initialize()
{
    if (uart.Connect()) {
        is_initialized = true;
        SPDLOG_INFO("UART connection established successfully.");
        return true;
    } else {
        SPDLOG_ERROR("Failed to establish UART connection.");
        return false;
    }
}


void ImageSender::Close()
{
    if (is_initialized) {
        uart.Disconnect();
        is_initialized = false;
        SPDLOG_INFO("UART connection closed.");
    }
}


bool ImageSender::SendImage(const std::string& image_path)
{
    if (!is_initialized) {
        SPDLOG_ERROR("UART not initialized. Call Initialize() before sending images.");
        return false;
    }

    // Perform handshake
    int32_t handshake_result = HandshakeWithMainboard();
    if (handshake_result != 1) {
        SPDLOG_ERROR("Handshake with mainboard failed with code: {}", handshake_result);
        return false; //TODO: implement retries
    }

    // Send image data
    uint32_t send_result = SendImageOverUart(image_path);
    if (send_result == 0) {
        SPDLOG_ERROR("Failed to send image over UART.");
        return false;
    }

    SPDLOG_INFO("Image sent successfully.");
    return true;
}


uint32_t ImageSender::HandshakeWithMainboard()
{
    // The main board initiates the handshake by sending a "START" message, and waits for an "SENDING" response.
    // then the orin sends an 'ACK' to confirm readiness.
    //After that, we start sending data image one by one, with each packet requiring an ACK from the main board 
    // before sending the next. if timeout, we get a nack and resend the packet.

    // Wait for "START" message from mainboard
    const std::string start_msg = "START";
    
    uint64_t start_time = timing::GetCurrentTimeMs();
    while (!shake_received){
        uint8_t cmd_id = static_cast<uint8_t>(CMD_HANDSHAKE_REQUEST); // Cast to uint8_t
        std::vector<uint8_t> data;

        bool bytes_received = uart.Receive(cmd_id, data);
        // bytes_received = true;

        if (bytes_received) {
            SPDLOG_INFO("Received : {}", bytes_received);
            // std::string received_msg(reinterpret_cast<char* >(data), strlen(reinterpret_cast<char*>(data)));
            std::string received_msg(data.begin(), data.end());
            if (received_msg == start_msg) {
                SPDLOG_INFO("Received START message from mainboard");
                shake_received = true;
                uart.Send(Packet::ToOut(std::vector<uint8_t>{'S','E','N','D','I','N','G'}));
                usleep(100000); // wait 100ms
                break;
            } else {
                SPDLOG_WARN("Received unexpected message: {}", received_msg);
            }
        } else {
            // SPDLOG_WARN("No message received from mainboard, retrying...");
            if (timing::GetCurrentTimeMs() - start_time > 30000) {
                return 2;
            }
        }
    }

    start_time = timing::GetCurrentTimeMs();
    const int MAX_HANDSHAKE_RETRIES = 5;
    while (!sending_ack_rec){
        uint8_t cmd_id = static_cast<uint8_t>(CMD_ACK_READY); // Cast to uint8_t
        std::vector<uint8_t> data;
        size_t bytes_received = uart.Receive(cmd_id, data); // 5 seconds timeout
        if (bytes_received > 0) {
            std::string response(data.begin(), data.end());
            if (response == "ACK") {
                SPDLOG_INFO("Handshake successful with mainboard");
                sending_ack_rec = true;
            } 
        } else if (timing::GetCurrentTimeMs() - start_time > 30000) { // 30 seconds timeout
            SPDLOG_ERROR("Timeout: No response received from mainboard during handshake");
            return 2;
        }
        
    }
    // read image file and construct and send packets while waiting for ACKs
    return 1; // Handshake successful
}

// Function to convert ImagePacket to Packet::Out
static inline Packet::Out convertToOut(const ImagePacket& packet) {
    Packet::Out out{};  // Zero-initialized std::array<uint8_t, 250>

    // Serialize the ImagePacket into a vector of bytes and then copy it into out
    size_t index = 0;
    out[index++] = static_cast<uint8_t>(packet.id);  // Add id as a byte
    out[index++] = static_cast<uint8_t>(packet.length); // Add length as a byte
    std::memcpy(out.data() + index, packet.data, packet.length); // Copy data[]
    index += packet.length;
    out[index] = packet.crc; // Add crc byte

    return out;
}



uint32_t ImageSender::SendImageOverUart(const std::string& image_path)
{
    std::vector<uint8_t> image_data;
    EC read_status = DH::ReadFileChunk(image_path, 0, DH::GetFileSize(image_path), image_data);

    if (read_status != EC::OK) {
        SPDLOG_ERROR("Failed to read image file: {}", image_path);
        return 0;
    }

    // Send image data over UART
    uint32_t bytes_sent = 0;
    uint32_t packet_id = 0;
    size_t image_data_size = image_data.size();
    uint64_t start_time;

    while (bytes_sent < image_data_size) {
        size_t chunk_size = std::min<size_t>(image_data_size - bytes_sent, MAX_PAYLOAD_SIZE);
        packet_id = bytes_sent / MAX_PAYLOAD_SIZE + 1;
        if (bytes_sent + chunk_size >= image_data_size){
            packet_id = 0; //last packet indicator 
        }
        Packet::Out packet = CreateImagePacket(packet_id, image_data.data() + bytes_sent, chunk_size);
        
        while (!uart.Send(packet)) {
            SPDLOG_INFO("Retrying sending image packet");
            // SPDLOG_ERROR("Failed to send image data chunk over UART");
            // return 0;
        }
        bytes_sent += chunk_size;
        // wait for ACK before continuing
        bool ack_received = false;
        start_time = timing::GetCurrentTimeMs();
        while (!ack_received){
            uint8_t cmd_id = static_cast<uint8_t>(CMD_ACK_OK); // Cast to uint8_t
            std::vector<uint8_t> data;
            bool bytes_received = uart.Receive(cmd_id, data); // 5 seconds timeout
            if (bytes_received) {
                std::string response(cmd_id, bytes_received);
                if (response == "ACK") {
                    SPDLOG_INFO("Received ACK for packet ID: {}", packet_id);
                    ack_received = true;
                } 
                else if (response == "NACK") {
                    uart.Send(packet); // Retry sending packet until ack is received
                }
            } else if (timing::GetCurrentTimeMs() - start_time > 10000) { // 10 seconds timeout
                SPDLOG_ERROR("Timeout: No ACK received for packet ID: {}", packet_id);
                return 0;
            }
        }
    }

    SPDLOG_INFO("Image sent successfully");
    return 1;
}



void ImageSender::RunImageTransfer() {
    ImageSender sender; //  "/dev/ttyTHS0", 115200??
    if (!sender.Initialize()) {
        SPDLOG_ERROR("Failed to initialize ImageSender");
        sender.Close();
        
        return;
    }
    // if (sender.HandshakeWithMainboard() != 1){
    //     SPDLOG_ERROR("Failed: Handshake not successful");
    //     sender.Close();

    //     return;
    // }
    if (!sender.SendImage("/home/argus/Documents/FSW-Payload/src/dog.jpeg")) {
        SPDLOG_ERROR("Failed to send image");
    }

    sender.Close();
}
