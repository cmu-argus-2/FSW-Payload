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
}

struct FileTransferManager
{
    // going against all rules..
    static inline std::mutex _mtx;
    static inline std::atomic<bool> _active_transfer = false;
    static inline uint16_t _total_seq_count = 0;
    static inline std::string _file_name = "";
    
    static bool active_transfer()
    {
        return _active_transfer.load();
    }

    static uint16_t total_seq_count()
    {
        std::lock_guard<std::mutex> lock(FileTransferManager::_mtx);
        return _total_seq_count;
    }

    static bool is_there_available_file()
    {
        return DH::CountFilesInDirectory(COMMS_FOLDER) > 0;
    }

    static EC populate_metadata(const std::string& file_name)
    {
        if (!is_there_available_file())
        {
            SPDLOG_INFO("No files available for transfer.");
            return EC::FILE_NOT_FOUND;
        }

        long file_size = DH::GetFileSize(file_name);
        if (file_size < 0)
        {
            SPDLOG_ERROR("Failed to get file size for {}.", file_name);
            return EC::FILE_DOES_NOT_EXIST;
        }

        // Calculate the total number of packets needed for transfer
        {
            std::lock_guard<std::mutex> lock(FileTransferManager::_mtx);
            _total_seq_count = static_cast<uint16_t>(std::ceil(static_cast<double>(file_size) / Packet::OUTGOING_PCKT_SIZE));
            SPDLOG_INFO("Total packets needed for transfer: {}", _total_seq_count);
            _file_name = file_name;
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