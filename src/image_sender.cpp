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
#include <string>
#include <ctime>
#include <map>
#include <any>


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

//TODO: MAKE SURE THIS WORKS
Packet::Out convert_image_packet_to_packet_out(std::vector<uint8_t> &image_packet_bytes) {
    Packet::Out output {};
    std::copy(image_packet_bytes.begin(), image_packet_bytes.end(), output.data());
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
std::vector<uint8_t> create_packet(
    uint8_t packet_id,
    uint8_t requested_packet = 0,
    const std::string& str_data = "",
    uint32_t chunk_id = 0,
    uint32_t data_length = 0,
    uint8_t last_packet = 0
) {
    std::vector<uint8_t> result;
    std::vector<uint8_t> data;

    if (!str_data.empty()) {
        data.assign(str_data.begin(), str_data.end());
        data_length = static_cast<uint32_t>(data.size());
    }

    if (packet_id == CMD_HANDSHAKE_REQUEST) {
        if (requested_packet == CMD_IMAGE_REQUEST) {
            result = {packet_id, requested_packet};
            result.resize(PACKET_SIZE, 0);
        }
    }
    else if (packet_id == CMD_DATA_CHUNK) {
        result.reserve(1 + 4 + 4 + data_length + 1);
        result.push_back(packet_id);

        // chunk_id (4 bytes, big endian)
        for (int i = 3; i >= 0; --i)
            result.push_back((chunk_id >> (8 * i)) & 0xFF);

        // data_length (4 bytes)
        for (int i = 3; i >= 0; --i)
            result.push_back((data_length >> (8 * i)) & 0xFF);

        // data
        result.insert(result.end(), data.begin(), data.end());

        // last_packet flag
        result.push_back(last_packet);

        uint8_t crc_result = create_crc5_packet(result, result.size());
        crc_result.resize(PACKET_SIZE, 0);
        result = crc_result;
    }
    else if (packet_id == CMD_ACK_OK || packet_id == CMD_NACK_CORRUPT) {
        result = {packet_id};
        for (int i = 3; i >= 0; --i)
            result.push_back((chunk_id >> (8 * i)) & 0xFF);
        result.resize(PACKET_SIZE, 0);
    }
    else if (packet_id == CMD_ACK_READY) {
        result = {packet_id, requested_packet};
        result.resize(PACKET_SIZE, 0);
    }

    return result;
}


std::map<std::string, std::any> read_packet(const std::vector<uint8_t>& bytes) {
    std::map<std::string, std::any> result;
    if (bytes.empty()) return result;

    uint8_t packet_id = bytes[0];
    result["packet_id"] = packet_id;

    if (packet_id == CMD_HANDSHAKE_REQUEST) {
        if (bytes.size() >= 2) {
            uint8_t requested_packet = bytes[1];
            std::string info(bytes.begin() + 2, bytes.end());
            // trim nulls
            info.erase(info.find_last_not_of('\0') + 1);
            result["requested_packet"] = requested_packet;
            result["data"] = info;
        }
    }

    else if (packet_id == CMD_DATA_CHUNK) {
        if (bytes.size() < 11) return result;  // 1 + 4 + 4 + 1 + 1 min size

        uint32_t chunk_id = (bytes[1] << 24) | (bytes[2] << 16) | (bytes[3] << 8) | bytes[4];
        uint32_t data_length = (bytes[5] << 24) | (bytes[6] << 16) | (bytes[7] << 8) | bytes[8];

        if (bytes.size() < 9 + data_length + 2) return result; // avoid overflow

        std::vector<uint8_t> data(bytes.begin() + 9, bytes.begin() + 9 + data_length);
        uint8_t last_packet = bytes[9 + data_length];
        uint8_t crc = bytes[10 + data_length];

        bool crc_valid = verify_crc5_packet(bytes);

        result["chunk_id"] = chunk_id;
        result["data_length"] = data_length;
        result["data"] = data;
        result["last_packet"] = last_packet;
        result["crc"] = crc;
        result["crc_valid"] = crc_valid;
    }

    else if (packet_id == CMD_ACK_READY) {
        if (bytes.size() >= 2) {
            uint8_t ready_packet_id = bytes[1];
            result["ready_packet_id"] = ready_packet_id;
        }
    }

    else if (packet_id == CMD_ACK_OK) {
        if (bytes.size() >= 5) {
            uint32_t acked_chunk_id =
                (bytes[1] << 24) | (bytes[2] << 16) | (bytes[3] << 8) | bytes[4];
            result["acked_chunk_id"] = acked_chunk_id;
        }
    }

    else if (packet_id == CMD_NACK_CORRUPT) {
        if (bytes.size() >= 5) {
            uint32_t failed_chunk_id =
                (bytes[1] << 24) | (bytes[2] << 16) | (bytes[3] << 8) | bytes[4];
            result["failed_chunk_id"] = failed_chunk_id;
        }
    }

    return result;
}

uint32_t ImageSender::HandshakeWithMainboard()
{
    uint64_t start_time = timing::GetCurrentTimeMs();
    uint8_t requested_pck_id;
    uint8_t received_pck_id;

    while (!shake_received){
        uint8_t cmd_id; // should be a buffer input will be stored in
        std::vector<uint8_t> data;

        bool bytes_received = uart.Receive(cmd_id, data);
        
        std::map<std::string, std::any> read_output = read_packet(data);

        if (bytes_received) {
            SPDLOG_INFO("Received : {}", bytes_received);
            // std::string received_msg(reinterpret_cast<char* >(data), strlen(reinterpret_cast<char*>(data)));
            received_pck_id = std::any_cast<uint8_t>(read_output["packet_id"]);

            if (received_pck_id == CMD_HANDSHAKE_REQUEST) {
                requested_pck_id = std::any_cast<uint8_t>(read_output["requested_packet"]);
                SPDLOG_INFO("Received START message from mainboard {}", received_pck_id);
                shake_received = true;
                ack_ready_to_send = create_packet(CMD_ACK_READY, CMD_IMAGE_REQUEST);
                uart.Send(Packet::ToOut(ack_ready_to_send));
                // usleep(100000); // wait 100ms
                break;
            } else {
                SPDLOG_WARN("Received unexpected message: {}", received_pck_id);
            }
        } else {
            // SPDLOG_WARN("No message received from mainboard, retrying...");
            if (timing::GetCurrentTimeMs() - start_time > 30000) {
                return 2;
            }
        }
    }
    received_pck_id = 0;
    start_time = timing::GetCurrentTimeMs();
    const int MAX_HANDSHAKE_RETRIES = 5;
    while (!sending_ack_rec){
        uint8_t cmd_id; 
        std::vector<uint8_t> data;
        bool bytes_received = uart.Receive(cmd_id, data); // 5 seconds timeout
        std::map<std::string, std::any> read_output = read_packet(data);

        if (bytes_received) {
            received_pck_id = std::any_cast<uint8_t>(read_output["packet_id"]);

            if (received_pck_id == CMD_ACK_READY) {
                requested_pck_id = std::any_cast<uint8_t>(read_output["ready_packet_id"]);
                SPDLOG_INFO("Handshake successful with mainboard going to send packet {}", requested_pck_id);
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
    uint32_t chunk_id = 0;
    size_t image_data_size = image_data.size();
    uint64_t start_time;
    uint8_t last_packet = 0;

    while (bytes_sent < image_data_size) {
        size_t chunk_size = std::min<size_t>(image_data_size - bytes_sent, MAX_PAYLOAD_SIZE);
        chunk_id = bytes_sent / MAX_PAYLOAD_SIZE + 1;
        if (bytes_sent + chunk_size >= image_data_size){
            last_packet = 1; //last packet indicator 
        }
        std::vector<uint8_t> image_packet_out = create_packet(
            CMD_DATA_CHUNK,
            0,
            "",
            chunk_id,
            static_cast<uint32_t>(chunk_size),
            last_packet
        );

        Packet::Out img_packet_send = convert_image_packet_to_packet_out(image_packet_out);
        uint64_t image_send_timeout = timing::GetCurrentTimeMs();
        while (!uart.Send(img_packet_send)) {
            SPDLOG_INFO("Retrying sending image packet");
            if (timing::GetCurrentTimeMs() - image_send_timeout > 10000) { // 10 seconds timeout
                SPDLOG_ERROR("Timeout: Failed to send image data chunk over UART");
                return 0;
            }
        }
        bytes_sent += chunk_size;
        // wait for ACK before continuing
        bool ack_received = false;
        start_time = timing::GetCurrentTimeMs();
        std::map<std::string, std::any> read_output;

        while (!ack_received){
            uint8_t cmd_id;
            std::vector<uint8_t> data;
            bool bytes_received = uart.Receive(cmd_id, data); // 5 seconds timeout
            if (bytes_received) {
                read_output = read_packet(data);
                uint8_t packet_id = std::any_cast<uint8_t>(read_output["packet_id"]);
                uint32_t acked_chunk_id = std::any_cast<uint32_t>(read_output["acked_chunk_id"]);
                if (packet_id == CMD_ACK_OK && acked_chunk_id == chunk_id) {
                    SPDLOG_INFO("Received ACK for chunk ID: {}", acked_chunk_id);
                    ack_received = true;
                } 
                else if (packet_id == CMD_ACK_NACK_CORRUPT) {
                    uart.Send(img_packet_send); // Retry sending packet until ack is received
                }
            } else if (timing::GetCurrentTimeMs() - start_time > 10000) { // 10 seconds timeout
                SPDLOG_ERROR("Timeout: No ACK received for chunk ID: {}", chunk_id);
                return 0;
            }
        }
    }

    SPDLOG_INFO("Image sent successfully");
    return 1;
}



// void ImageSender::RunImageTransfer() {
//     ImageSender sender; //  "/dev/ttyTHS0", 115200??
//     if (!sender.Initialize()) {
//         SPDLOG_ERROR("Failed to initialize ImageSender");
//         sender.Close();
        
//         return;
//     }
//     // if (sender.HandshakeWithMainboard() != 1){
//     //     SPDLOG_ERROR("Failed: Handshake not successful");
//     //     sender.Close();

//     //     return;
//     // }
//     if (!sender.SendImage("/home/argus/Documents/FSW-Payload/src/dog.jpeg")) {
//         SPDLOG_ERROR("Failed to send image");
//     }

//     sender.Close();
// }





void ImageSender::RunImageTransfer() {
    UART uart_curr; //  "/dev/ttyTHS0", 115200??
    if (!uart_curr.Connect()) {
        SPDLOG_ERROR("Failed to initialize ImageSender");
        uart_curr.Disconnect();

        return;
    }
    while (true){
        uart_curr.Send(Packet::ToOut(std::vector<uint8_t>{'N', 'I', 'C', 'E'}));
        usleep(1000000); // wait 1s
    }
    uart_curr.Disconnect();
}

// void ImageSender::RunImageTransfer() {
//     ImageSender sender; //  "/dev/ttyTHS0", 115200??
//     if (!sender.Initialize()) {
//         SPDLOG_ERROR("Failed to initialize ImageSender");
//         sender.Close();
        
//         return;
//     }
//     while (true){
            
//         // uint8_t cmd_id = static_cast<uint8_t>(CMD_HANDSHAKE_REQUEST); // Cast to uint8_t
//         uint8_t & cmd_id; // should be a buffer input will be stored in
//         std::vector<uint8_t> data;

//         bool bytes_received = uart.Receive(cmd_id, data);
//         // bytes_received = true;

//         if (bytes_received) {
//             SPDLOG_INFO("Received : {}", bytes_received);
//             // std::string received_msg(reinterpret_cast<char* >(data), strlen(reinterpret_cast<char*>(data)));
//             std::string received_msg(data.begin(), data.end());
//             if (received_msg == start_msg) {
//                 SPDLOG_INFO("Received START message from mainboard");
//                 shake_received = true;
//                 uart.Send(Packet::ToOut(std::vector<uint8_t>{'S','E','N','D','I','N','G'}));
//                 usleep(100000); // wait 100ms
//                 break;
//             } else {
//                 SPDLOG_WARN("Received unexpected message: {}", received_msg);
//             }
//         } else {
//             // SPDLOG_WARN("No message received from mainboard, retrying...");
//             if (timing::GetCurrentTimeMs() - start_time > 30000) {
//                 return 2;
//             }
//         }
//     }
//     sender.Close();
// }

