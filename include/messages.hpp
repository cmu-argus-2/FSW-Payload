#ifndef MESSAGES_HPP
#define MESSAGES_HPP

#include <vector>
#include <cstdint>
#include <chrono>
#include "commands.hpp"

// Base message structure
struct Message {
    uint8_t id;               // ID field (1 byte)
    uint16_t seq_count = 1;       // Sequence count (2 bytes), default to 1
    uint8_t data_length;      // Data length field (1 byte)

    std::vector<uint8_t> packet = {}; // Serialized packet buffer
    
    uint8_t priority = 1;         // For the TX priority queue
    std::chrono::system_clock::time_point created_at; // Timestamp for when the task was created.
    
    Message(uint8_t id, uint8_t data_length, uint16_t seq_count = 1);

    virtual ~Message() = default; // Virtual destructor
    virtual void serialize() = 0; // Pure virtual method for serialization
};

struct MSG_RequestState : public Message {
    uint8_t state;

    MSG_RequestState();
    void serialize() override; 
};

#endif // MESSAGES_HPP
