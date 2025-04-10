#ifndef COMMS_HPP
#define COMMS_HPP

#include <vector>
#include <cstdint>
#include <atomic>
#include <string_view>

#include "spdlog/spdlog.h"
#include "core/data_handling.hpp"

// Asymmetric sizes for send and receive buffers
namespace Packet
{
    static constexpr uint8_t OUTGOING_PCKT_SIZE = 250;
    static constexpr uint8_t INCOMING_PCKT_SIZE = 32;

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

        // Calculate the total number of packets needed for transfer
        {
            std::lock_guard<std::mutex> lock(FileTransferManager::_mtx);
            _total_seq_count = static_cast<uint16_t>(std::ceil(static_cast<double>(file_size) / Packet::OUTGOING_PCKT_SIZE));
            SPDLOG_INFO("Total packets needed for transfer: {}", _total_seq_count);
            _file_path = file_path;
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

        // Read the file chunk
        EC err = DH::ReadFileChunk(_file_path, seq_number * Packet::OUTGOING_PCKT_SIZE, Packet::OUTGOING_PCKT_SIZE, data);
        if (err != EC::OK)
        {
            LogError(EC::FAILED_TO_GRAB_FILE_CHUNK);
            return EC::FAILED_TO_GRAB_FILE_CHUNK;
        }

        return EC::OK;
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
    virtual bool Send(const std::vector<uint8_t>& data) = 0;
    virtual void RunLoop() = 0;
    virtual void StopLoop() = 0;
    virtual bool IsConnected() const { return _connected; }
    

protected:

    bool _connected;
    std::atomic<bool> _running_loop;

};


#endif // COMMS_HPP