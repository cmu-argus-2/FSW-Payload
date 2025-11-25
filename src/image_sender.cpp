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



// Function to calculate CRC5 for a full block of data (e.g., 8 bytes)
uint8_t calculate_crc5(const std::vector<uint8_t>& data_bytes, size_t length) {
    const uint8_t polynomial = 0x05;  // CRC5 polynomial (x^5 + x^2 + 1)
    uint8_t crc = 0x1F; 
    uint8_t num_bytes = PACKET_SIZE; 
    uint8_t num_bits = num_bytes * 8; 

    // Process each byte in the data array
    for (size_t i = 0; i < data_bytes.size(); ++i) {
        uint8_t byte = data_bytes[i];
        
        // Process each bit in the byte
        for (int bit = 7; bit >= 0; --bit) {
            if ((crc & 0x10) ^ ((byte >> bit) & 0x01)) {
                crc = (crc << 1) ^ polynomial;
            } else {
                crc = (crc << 1);
            }
            crc &= 0x1F;
        }
    }

    return crc;
}

// Convert image packet vector to Packet::Out
Packet::Out convert_image_packet_to_packet_out(std::vector<uint8_t> &image_packet_bytes) {
    Packet::Out output {};
    std::copy(image_packet_bytes.begin(), image_packet_bytes.end(), output.data());
    return output;
}


ImageSender::ImageSender()
    : uart(), is_initialized(false)
{

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
void write_result_to_file(const std::vector<uint8_t>& data, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        SPDLOG_ERROR("Failed to open file {} for writing.", filename);
        return;
    }
    outfile.write(reinterpret_cast<const char*>(data.data()), data.size());
    outfile.close();
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
        result.reserve(1 + 4 + 4 + data_length + 1); // packet_id + chunk_id + data_length + data + last_packet + crc5  
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
        
        uint8_t crc_result = calculate_crc5(result, result.size());
        // SPDLOG_WARN("chunkid: {}   CRC calculated: {}", chunk_id, crc_result);
        result.push_back(crc_result);
        result.resize(PACKET_SIZE, 0);
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

        bool crc_valid = (crc == calculate_crc5(bytes, 10 + data_length));

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
                std::vector<uint8_t> ack_ready_to_send = create_packet(CMD_ACK_READY, CMD_IMAGE_REQUEST);
                uart.Send(Packet::ToOut(std::move(ack_ready_to_send)));
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

std::vector<uint8_t> read_jpeg_to_bytes(const std::string& filename) {
    // Open the JPEG file in binary mode
    std::ifstream file(filename, std::ios::binary);
    
    // Read the file into a vector
    std::vector<uint8_t> file_bytes((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());
    
    return file_bytes;
}

std::vector<uint8_t> read_binary_file(const std::string& filename) {
    // Open the binary file for reading
    std::ifstream file(filename, std::ios::binary);

    // Read the file into a vector of bytes
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
    file.close();
    return data;
}

// Create a dictionary of data packets for a requested image binary file
std::map<uint32_t, std::vector<uint8_t>> ImageSender::create_packet_dict(const std::string& image_path) {
    std::map<uint32_t, std::vector<uint8_t>> packet_dict;
    std::vector<uint8_t> image_data = read_binary_file(image_path);
    if (image_data.empty()) {
        SPDLOG_ERROR("Failed to read image file: {}", image_path);
        return packet_dict;
    }   
    uint32_t bytes_sent = 0;
    uint32_t chunk_id = 0;
    size_t image_data_size = image_data.size();
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
            std::string(image_data.begin() + bytes_sent, image_data.begin() + bytes_sent + chunk_size),
            chunk_id,
            static_cast<uint32_t>(chunk_size),
            last_packet
        );

        packet_dict[chunk_id] = image_packet_out;
        bytes_sent += chunk_size;
    }
    return packet_dict;
}


uint32_t ImageSender::SendImageOverUart(const std::string& image_path)
{
    // Send image data over UART
    uint32_t bytes_sent = 0;
    uint32_t chunk_id = 0;
    uint64_t start_time;
    std::map<uint32_t, std::vector<uint8_t>> packet_dict = create_packet_dict(image_path);
    
    // Loop throught the packet dict and send each packet: Wait for ack before sending next packet if nack resend
    for (const auto& [chunk_id, image_packet_out] : packet_dict) {
        SPDLOG_WARN("Sending chunk ID: {}, size: {}", chunk_id, image_packet_out.size());
        Packet::Out img_packet_send = convert_image_packet_to_packet_out(const_cast<std::vector<uint8_t>&>(image_packet_out));
        uint64_t image_send_timeout = timing::GetCurrentTimeMs();
        while (!uart.Send(img_packet_send)) {
            SPDLOG_INFO("Retrying sending image packet");
            if (timing::GetCurrentTimeMs() - image_send_timeout > 10000) { // 10 seconds timeout
                SPDLOG_ERROR("Timeout: Failed to send image data chunk over UART");
                return 0;
            }
        }
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
                if (packet_id == CMD_ACK_OK){
                    SPDLOG_INFO("Packet ID: {}", packet_id);
                    uint32_t acked_chunk_id = std::any_cast<uint32_t>(read_output["acked_chunk_id"]);
                    SPDLOG_INFO("Received ACK for chunk ID: {}", acked_chunk_id);
                    ack_received = true;
                }else if (packet_id == CMD_NACK_CORRUPT) {
                    uint32_t failed_chunk_id = std::any_cast<uint32_t>(read_output["failed_chunk_id"]);
                    SPDLOG_WARN("RECEIVED NACK FOR CHUNK ID: {}", failed_chunk_id);

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



void ImageSender::RunImageTransfer() {
    ImageSender sender; //  "/dev/ttyTHS0", 115200??
    if (!sender.Initialize()) {
        SPDLOG_ERROR("Failed to initialize ImageSender");
        sender.Close();
        
        return;
    }
    if (sender.SendImage("/home/argus/Desktop/image_transfer/FSW-Payload/src/image_radio_file.bin")) {
        SPDLOG_INFO("BIN transfer completed successfully.");
    } else {
        SPDLOG_ERROR("BIN transfer failed.");
    }

    sender.Close();
}

