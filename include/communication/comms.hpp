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
    // Packet header structure for binary files
    struct PacketInfo {
        size_t offset;      // byte offset in file
        uint16_t payload_size;  // payload size from header
    };

    // going against all rules..
    static inline std::mutex _mtx;
    static inline std::atomic<bool> _active_transfer = false;
    static inline uint16_t _total_seq_count = 0;
    static inline std::string _file_path = "";
    static inline std::vector<PacketInfo> _packet_offsets;
    static inline bool _is_binary_packet_file = false;
    
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
        if (!IsThereAvailableFile())
        {
            SPDLOG_INFO("No files available for transfer.");
            return EC::FILE_NOT_FOUND;
        }

        long file_size = DH::GetFileSize(file_path);
        if (file_size < 0)
        {
            SPDLOG_ERROR("Failed to get file size for {}.", file_path);
            return EC::FILE_DOES_NOT_EXIST;
        }

        std::lock_guard<std::mutex> lock(FileTransferManager::_mtx);
        _file_path = file_path;
        _packet_offsets.clear();
        
        // Check if this is a binary packet file (img_*.bin)
        _is_binary_packet_file = (file_path.find("img_") != std::string::npos && 
                                   file_path.substr(file_path.length() - 4) == ".bin");

        if (_is_binary_packet_file)
        {
            // Parse the binary file to find all packet headers
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open())
            {
                SPDLOG_ERROR("Failed to open file for packet parsing: {}", file_path);
                return EC::FILE_NOT_FOUND;
            }

            size_t current_offset = 0;
            while (current_offset < static_cast<size_t>(file_size))
            {
                file.seekg(current_offset);
                
                // Read 7-byte packet header
                uint8_t header[7];
                file.read(reinterpret_cast<char*>(header), 7);
                if (file.gcount() != 7)
                {
                    SPDLOG_WARN("Incomplete packet header at offset {}", current_offset);
                    break;
                }

                // Parse payload size
                uint16_t payload_size = (static_cast<uint16_t>(header[0]) << 8) | header[1];
                
                PacketInfo info;
                info.offset = current_offset;
                info.payload_size = payload_size;
                _packet_offsets.push_back(info);

                SPDLOG_DEBUG("Packet {}: offset={}, payload_size={}", _packet_offsets.size() - 1, current_offset, payload_size);

                // Move to next packet (7-byte header + payload)
                current_offset += 7 + payload_size;
            }

            _total_seq_count = static_cast<uint16_t>(_packet_offsets.size());
            SPDLOG_INFO("Found {} packets in binary file", _total_seq_count);
        }
        else
        {
            // Regular file - calculate packets based on MAX_DATA_LENGTH
            _total_seq_count = static_cast<uint16_t>(std::ceil(static_cast<double>(file_size) / Packet::MAX_DATA_LENGTH));
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
            _packet_offsets.clear();
            _is_binary_packet_file = false;
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

        if (_is_binary_packet_file)
        {
            // Read the specific packet (header + payload) from binary file
            if (seq_number >= _packet_offsets.size())
            {
                SPDLOG_ERROR("Packet index {} out of range", seq_number);
                return EC::NO_MORE_PACKET_FOR_FILE;
            }

            const PacketInfo& packet_info = _packet_offsets[seq_number];
            size_t packet_total_size = 7 + packet_info.payload_size; // header + payload

            std::ifstream file(_file_path, std::ios::binary);
            if (!file.is_open())
            {
                SPDLOG_ERROR("Failed to open file for reading packet");
                return EC::FILE_NOT_FOUND;
            }

            file.seekg(packet_info.offset);
            data.resize(packet_total_size);
            file.read(reinterpret_cast<char*>(data.data()), packet_total_size);
            
            if (file.gcount() != static_cast<std::streamsize>(packet_total_size))
            {
                SPDLOG_ERROR("Failed to read complete packet {} (expected {} bytes, got {})", 
                            seq_number, packet_total_size, file.gcount());
                return EC::FAILED_TO_GRAB_FILE_CHUNK;
            }

            SPDLOG_DEBUG("Read packet {}: {} bytes (7-byte header + {}-byte payload)", 
                        seq_number, packet_total_size, packet_info.payload_size);
            return EC::OK;
        }
        else
        {
            // Regular file - read by MAX_DATA_LENGTH chunks
            EC err = DH::ReadFileChunk(_file_path, seq_number * Packet::MAX_DATA_LENGTH, Packet::MAX_DATA_LENGTH, data);
            if (err != EC::OK)
            {
                LogError(EC::FAILED_TO_GRAB_FILE_CHUNK);
                return EC::FAILED_TO_GRAB_FILE_CHUNK;
            }
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