#include "messages.hpp"
#include <cassert>

Message::Message(uint8_t id, uint8_t data_length, uint16_t seq_count) 
    : id(id), seq_count(seq_count), data_length(data_length), _serialized(false)
{
    created_at = std::chrono::system_clock::now();

    if (seq_count > 1) {
        priority = TX_PRIORITY_2;
    }

    packet.reserve(data_length + 4); // Reserve space for the header and data
    SerializeHeader();
}

void Message::SerializeHeader()
{
    packet.clear();
    packet.push_back(id); 
    packet.push_back(static_cast<uint8_t>(seq_count >> 8));
    packet.push_back(static_cast<uint8_t>(seq_count & 0xFF));
    packet.push_back(data_length);
}



void Message::AddToPacket(std::vector<uint8_t>& data)
{
    packet.insert(packet.end(), data.begin(), data.end());
}

void Message::AddToPacket(uint8_t data)
{
    packet.push_back(data);
}

bool Message::VerifyPacketSerialization()
{
    _serialized = (packet.size() == data_length + 4);
    return _serialized;
}

std::shared_ptr<Message> CreateMessage(CommandID::Type id, std::vector<uint8_t>& tx_data)
{
    auto msg = std::make_shared<Message>(id, tx_data.size());
    msg->AddToPacket(tx_data);

    assert(msg->VerifyPacketSerialization() && ("Packet serialization verification failed for CommandID: " + std::to_string(static_cast<int>(id))).c_str());
    
    return msg;
}

void SerializeToBytes(uint32_t value, std::vector<uint8_t>& output)
{
    output.push_back(static_cast<uint8_t>(value >> 24));
    output.push_back(static_cast<uint8_t>(value >> 16));
    output.push_back(static_cast<uint8_t>(value >> 8));
    output.push_back(static_cast<uint8_t>(value & 0xFF));
}

void SerializeToBytes(uint16_t value, std::vector<uint8_t>& output)
{
    output.push_back(static_cast<uint8_t>(value >> 8));
    output.push_back(static_cast<uint8_t>(value & 0xFF));
}

std::shared_ptr<Messgae> CreateSuccessAckMessage(CommandID::Type id)
{
    auto msg = std::make_shared<Message>(id, 1);
    msg->AddToPacket(SUCCESS_FLAG);

    assert(msg->VerifyPacketSerialization() && ("Packet serialization verification failed for CommandID: " + std::to_string(static_cast<int>(id))).c_str());

    return msg;
}


std::shared_ptr<Messgae> CreateErrorAckMessage(CommandID::Type id, uint8_t error_code)
{
    auto msg = std::make_shared<Message>(id, 2);
    msg->AddToPacket(ERROR_FLAG);
    msg->AddToPacket(error_code);

    assert(msg->VerifyPacketSerialization() && ("Packet serialization verification failed for CommandID: " + std::to_string(static_cast<int>(id))).c_str());

    return msg;
}