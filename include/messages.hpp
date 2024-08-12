#ifndef MESSAGES_HPP
#define MESSAGES_HPP

#include <vector>
#include <cstdint>

// Base message structure
struct BaseMessage {
    uint8_t id;               // ID field (1 byte)
    uint16_t seq_count;       // Sequence count (2 bytes), default to 1
    uint8_t data_length;      // Data length field (1 byte)
    
    BaseMessage(uint8_t id, uint8_t data_length, uint16_t seq_count = 1) 
        : id(id), seq_count(seq_count), data_length(data_length) {}

    virtual ~BaseMessage() = default; // Virtual destructor
    virtual void serialize(std::vector<uint8_t>& buffer) const = 0; // Pure virtual method for serialization
};


struct RequestStateMessage : public BaseMessage {
    uint8_t state;

    RequestStateMessage(); 
    void serialize(std::vector<uint8_t>& buffer) const override; 
};

#endif // MESSAGES_HPP
