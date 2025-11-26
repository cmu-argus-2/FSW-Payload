#ifndef MESSAGES_HPP
#define MESSAGES_HPP

#include <vector>
#include <cstdint>
#include <chrono>
#include <atomic>
#include <memory>
#include <communication/comms.hpp>
#include "commands.hpp"

// If seq_count > 1, priority is automatically set to 2
#define TX_PRIORITY_1 1
#define TX_PRIORITY_2 2

#define ACK_SUCCESS 0x0A
#define ACK_ERROR 0x0B

// Base message structure
struct Message 
{
    uint8_t id;               // ID field (1 byte)
    uint16_t seq_count = 1;       // Sequence count (2 bytes), default to 1
    uint8_t data_length;      // Data length field (1 byte)

    std::atomic<bool> _serialized = false; // Atomic flag for serialization status  

    std::vector<uint8_t> packet = {}; // Serialized packet buffer
    Packet::Out _packet{}; // Serialized packet buffer - will switch soon to this one
    
    uint8_t priority = TX_PRIORITY_1;         // For the TX priority queue
    uint64_t created_at; // UNIX timestamp for when the task was created.
    
    Message(uint8_t id, uint8_t data_length, uint16_t seq_count = 1);

    virtual ~Message() = default; // Virtual destructor
    void AddToPacket(std::vector<uint8_t>& data); // Add data (vector) to the packet
    void AddToPacket(uint8_t data); // Add data to the packet
    void SerializeHeader(); // Serialize the header
    bool VerifyPacketSerialization(); // Check if the packet is serialized
    bool Serialized() const { return _serialized; } // Check if the message has been serialized
};


std::shared_ptr<Message> CreateMessage(CommandID::Type id, std::vector<uint8_t>& tx_data, uint16_t seq_count = 1);
void SerializeToBytes(uint64_t value, std::vector<uint8_t>& output);
void SerializeToBytes(uint32_t value, std::vector<uint8_t>& output);
void SerializeToBytes(uint16_t value, std::vector<uint8_t>& output);

std::shared_ptr<Message> CreateSuccessAckMessage(CommandID::Type id);
std::shared_ptr<Message> CreateErrorAckMessage(CommandID::Type id, uint8_t error_code);

// CRC16-CCITT functions
uint16_t calculate_crc16(const uint8_t* data, size_t length);
bool verify_crc16(const uint8_t* data, size_t length);


#endif // MESSAGES_HPP
