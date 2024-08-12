#include "messages.hpp"


void MSG_RequestState::serialize(std::vector<uint8_t>& buffer) const {
    buffer.push_back(id); 
    buffer.push_back(static_cast<uint8_t>(seq_count >> 8));
    buffer.push_back(static_cast<uint8_t>(seq_count & 0xFF));
    buffer.push_back(data_length);
    buffer.push_back(state); 
}
