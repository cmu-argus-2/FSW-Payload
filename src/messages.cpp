#include "messages.hpp"


Message::Message(uint8_t id, uint8_t data_length, uint16_t seq_count) 
    : id(id), seq_count(seq_count), data_length(data_length) 
{
    created_at = std::chrono::system_clock::now();

    if (seq_count > 1) {
        priority = TX_PRIORITY_2;
    }
}

MSG_PING_ACK::MSG_PING_ACK() 
    : Message(CommandID::PING_ACK, 1) {}

void MSG_PING_ACK::serialize()
{
    packet.clear();
    packet.push_back(id); 
    packet.push_back(static_cast<uint8_t>(seq_count >> 8));
    packet.push_back(static_cast<uint8_t>(seq_count & 0xFF));
    packet.push_back(data_length);

    packet.push_back(0x60); // Ping
}

