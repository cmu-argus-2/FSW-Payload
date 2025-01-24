#ifndef MESSAGES_HPP
#define MESSAGES_HPP

#include <vector>
#include <cstdint>
#include <chrono>
#include <atomic>
#include "commands.hpp"

// If seq_count > 1, priority is automatically set to 2
#define TX_PRIORITY_1 1
#define TX_PRIORITY_2 2


// Base message structure
struct Message 
{
    uint8_t id;               // ID field (1 byte)
    uint16_t seq_count = 1;       // Sequence count (2 bytes), default to 1
    uint8_t data_length;      // Data length field (1 byte)

    std::atomic<bool> _serialized = false; // Atomic flag for serialization status  

    std::vector<uint8_t> packet = {}; // Serialized packet buffer
    
    uint8_t priority = TX_PRIORITY_1;         // For the TX priority queue
    std::chrono::system_clock::time_point created_at; // Timestamp for when the task was created.
    
    Message(uint8_t id, uint8_t data_length, uint16_t seq_count = 1);

    virtual ~Message() = default; // Virtual destructor
    virtual void serialize() = 0; // Pure virtual method for serialization
    virtual bool Serialized() const { return _serialized; } // Check if the message has been serialized
};

struct MSG_PING_ACK : public Message 
{
    MSG_PING_ACK();
    void serialize() override; 
};

#endif // MESSAGES_HPP
