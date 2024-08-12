#include "messages.hpp"


Message::Message(uint8_t id, uint8_t data_length, uint16_t seq_count) 
    : id(id), seq_count(seq_count), data_length(data_length) 
{
    created_at = std::chrono::system_clock::now();
}


MSG_RequestState::MSG_RequestState() 
    : Message(CommandID::REQUEST_STATE, 1) {}

void MSG_RequestState::serialize(){
    packet.push_back(id); 
    packet.push_back(static_cast<uint8_t>(seq_count >> 8));
    packet.push_back(static_cast<uint8_t>(seq_count & 0xFF));
    packet.push_back(data_length);
    packet.push_back(state); 
}
