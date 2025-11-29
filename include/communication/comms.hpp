#ifndef COMMS_HPP
#define COMMS_HPP

#include <vector>
#include <cstdint>
#include <atomic>
#include <string_view>
#include <fstream>

#include "spdlog/spdlog.h"
#include "core/data_handling.hpp"

// Asymmetric sizes for send and receive buffers
namespace Packet
{
    static constexpr uint8_t OUTGOING_PCKT_SIZE = 246;  // 1B cmd_id + 2B seq_count + 1B data_len + 240B data + 2B CRC16
    static constexpr uint8_t INCOMING_PCKT_SIZE = 32;

    static constexpr uint8_t MAX_DATA_LENGTH = 240;  // Maximum payload data (before CRC16)
    static constexpr uint8_t FILE_CHUNK_DATA_LEN = MAX_DATA_LENGTH; // send up to 240 raw bytes per frame

    using Out = std::array<uint8_t, OUTGOING_PCKT_SIZE>;
    using In = std::array<uint8_t, INCOMING_PCKT_SIZE>;

    // TODO: Temporary conversion copying std::vector content into the array until I fully switch to this container
    static inline Out ToOut(std::vector<uint8_t>&& vec) 
    {
        Out out{}; // zero-initialized
        std::move(vec.begin(), vec.begin() + std::min(vec.size(), static_cast<size_t>(OUTGOING_PCKT_SIZE)), out.begin());
        return out;
    }

    static inline In ToIn(std::vector<uint8_t>&& vec) 
    {
        In in{}; // zero-initialized
        std::move(vec.begin(), vec.begin() + std::min(vec.size(), static_cast<size_t>(INCOMING_PCKT_SIZE)), in.begin());
        return in;
    }
}

struct FileTransferManager
{
    // going against all rules..
    static inline std::mutex _mtx;
    static inline std::atomic<bool> _active_transfer = false;
    static inline uint16_t _total_seq_count = 0;
    static inline std::string _file_path = "";
    
    static bool active_transfer()
    {
        return _active_transfer.load();
    }

    static uint16_t total_seq_count()
    {
        std::lock_guard<std::mutex> lock(FileTransferManager::_mtx);
        return _total_seq_count;
    }

    static bool IsThereAvailableFile()
    {
        return DH::CountFilesInDirectory(COMMS_FOLDER) > 0;
    }

    static EC PopulateMetadata(const std::string& file_path)
    {
        // Check if the specific file exists
        long file_size = DH::GetFileSize(file_path);
        if (file_size < 0)
        {
            SPDLOG_ERROR("Failed to get file size for {}.", file_path);
            return EC::FILE_DOES_NOT_EXIST;
        }

        std::lock_guard<std::mutex> lock(FileTransferManager::_mtx);
        _file_path = file_path;
        // Detect DH magic to compute packet count by fixed record size (5-byte magic + 242-byte records)
        bool is_dh = false;
        {
            std::ifstream file(file_path, std::ios::binary);
            if (file.is_open()) {
                char magic[5];
                file.read(magic, 5);
                is_dh = file.gcount() == 5 && std::memcmp(magic, "DHGEN", 5) == 0;
            }
        }

        if (is_dh) {
            // File structure: 5B magic + N * 242B records (2B len + payload + padding)
            const std::size_t record_size = 242;
            if (file_size < 5) {
                SPDLOG_ERROR("DH-formatted file too small: {}", file_size);
                return EC::FILE_DOES_NOT_EXIST;
            }
            std::size_t payload_bytes = static_cast<std::size_t>(file_size - 5);
            _total_seq_count = static_cast<uint16_t>(payload_bytes / record_size);
            SPDLOG_INFO("Detected DH-formatted file. Records: {}", _total_seq_count);
        } else {
            // Treat all other files the same: chunk into FILE_CHUNK_DATA_LEN slices
            _total_seq_count = static_cast<uint16_t>(std::ceil(static_cast<double>(file_size) / Packet::FILE_CHUNK_DATA_LEN));
            SPDLOG_INFO("Total packets needed for transfer: {}", _total_seq_count);
        }

        _active_transfer.store(true);
        return EC::OK;
    }

    static void Reset()
    {
        _active_transfer.store(false);
        {
            std::lock_guard<std::mutex> lock(FileTransferManager::_mtx);
            _total_seq_count = 0;
        }
    }

    static EC GrabFileChunk(uint16_t seq_number, std::vector<uint8_t>& data)
    {
        std::lock_guard<std::mutex> lock(FileTransferManager::_mtx);
        if (seq_number > _total_seq_count)
        {
            SPDLOG_ERROR("Requested packet number {} is out of range.", seq_number);
            return EC::NO_MORE_PACKET_FOR_FILE;
        }

        // Detect DH magic to decide how to slice
        bool is_dh = false;
        {
            std::ifstream file(_file_path, std::ios::binary);
            if (file.is_open()) {
                char magic[5];
                file.read(magic, 5);
                is_dh = file.gcount() == 5 && std::memcmp(magic, "DHGEN", 5) == 0;
            }
        }

        if (is_dh) {
            // DH file: Extract payload from the 242-byte record
            // Each record is 242 bytes: [2B length][up to 240B data][padding]
            const std::size_t record_size = 242;
            std::size_t offset = 5 + static_cast<std::size_t>(seq_number - 1) * record_size;
            
            // Read the full 242-byte record
            std::vector<uint8_t> dh_packet;
            EC err = DH::ReadFileChunk(_file_path, static_cast<uint32_t>(offset), static_cast<uint32_t>(record_size), dh_packet);
            if (err != EC::OK || dh_packet.size() != record_size) {
                SPDLOG_ERROR("Failed to read DH record at packet {}", seq_number);
                LogError(EC::FAILED_TO_GRAB_FILE_CHUNK);
                return EC::FAILED_TO_GRAB_FILE_CHUNK;
            }
            
            // Extract payload length from 2-byte header (big-endian)
            const uint16_t payload_len = (static_cast<uint16_t>(dh_packet[0]) << 8) | dh_packet[1];
            if (payload_len > 240) {
                SPDLOG_ERROR("Invalid DH record length {} at packet {}", payload_len, seq_number);
                return EC::FAILED_TO_GRAB_FILE_CHUNK;
            }
            
            // Extract only the actual payload (skip 2-byte length header)
            data.clear();
            data.reserve(payload_len);
            data.insert(data.end(), dh_packet.begin() + 2, dh_packet.begin() + 2 + payload_len);
            
            SPDLOG_DEBUG("Extracted payload from DH packet {}: {} bytes (from 242-byte record)", seq_number, payload_len);
            return EC::OK;
        } else {
            // Non-DH file: Read raw data in 240-byte chunks
            uint32_t offset = (seq_number == 0) ? 0 : static_cast<uint32_t>((seq_number - 1) * Packet::FILE_CHUNK_DATA_LEN);
            EC err = DH::ReadFileChunk(_file_path, offset, Packet::FILE_CHUNK_DATA_LEN, data);
            if (err != EC::OK)
            {
                LogError(EC::FAILED_TO_GRAB_FILE_CHUNK);
                return EC::FAILED_TO_GRAB_FILE_CHUNK;
            }
            SPDLOG_DEBUG("Read raw file chunk {}: {} bytes", seq_number, data.size());
            return EC::OK;
        }
    }
};

// Abstract class for communication interfaces
class Communication
{

public:

    Communication()
        : _connected(false), _running_loop(false) {}
    
    virtual ~Communication() = default; // Ensure proper cleanup for derived classes


    virtual bool Connect() = 0;
    virtual void Disconnect() = 0;
    virtual bool Receive(uint8_t& cmd_id, std::vector<uint8_t>& data) = 0;
    virtual bool Send(const Packet::Out& data) = 0;
    virtual void RunLoop() = 0;
    virtual void StopLoop() = 0;
    virtual bool IsConnected() const { return _connected; }
    

protected:

    bool _connected;
    std::atomic<bool> _running_loop;

};

#endif // COMMS_HPP
