#include "messages.hpp"
#include <cassert>
#include "core/timing.hpp"

Message::Message(uint8_t id, uint16_t data_length, uint16_t seq_count) 
    : 
    id(id), 
    seq_count(seq_count), 
    data_length(data_length),
     _serialized(false)
{
    created_at = timing::GetCurrentTimeMs();

    if (seq_count > 1) 
    {
        priority = TX_PRIORITY_2;
    }

    packet.reserve(data_length + 5); // Reserve space for the header (5 bytes now) and data
    SerializeHeader();
}

void Message::SerializeHeader()
{
    packet.clear();
    packet.push_back(id); 
    packet.push_back(static_cast<uint8_t>(seq_count >> 8));
    packet.push_back(static_cast<uint8_t>(seq_count & 0xFF));
    packet.push_back(static_cast<uint8_t>(data_length >> 8));  // 2-byte length high byte
    packet.push_back(static_cast<uint8_t>(data_length & 0xFF)); // 2-byte length low byte
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
    // ACKs: 6 bytes (5 header + 1 status, NO CRC)
    // Data packets: 247 bytes (5 header + 240 data + 2 CRC)
    if (data_length == 1) {
        _serialized = (packet.size() == 6);  // ACK packet
    } else {
        _serialized = (packet.size() == 247);  // Data packet
    }
    return _serialized;
}

std::shared_ptr<Message> CreateMessage(CommandID::Type id, std::vector<uint8_t>& tx_data, uint16_t seq_count)
{
    // Data packets are always 247 bytes: 5 header + 240 data (padded) + 2 CRC
    // tx_data contains the actual payload (≤240 bytes)
    uint16_t header_data_length = static_cast<uint16_t>(tx_data.size());
    
    if (header_data_length > 240) {
        throw std::runtime_error("Payload too large: " + std::to_string(header_data_length) + " bytes (max 240)");
    }
    
    auto msg = std::make_shared<Message>(id, header_data_length, seq_count);
    msg->AddToPacket(tx_data);

    // Always pad to 240 bytes total data for fixed-size 247-byte packets
    size_t current_size = msg->packet.size();  // 5 (header) + actual_data_size
    size_t target_size = 245;  // 5 (header) + 240 (padded data)
    
    if (current_size < target_size) {
        msg->packet.resize(target_size, 0);  // Pad with zeros
    }

    // Calculate CRC16 over [header + padded data] (bytes 0-244)
    uint16_t crc = calculate_crc16(msg->packet.data(), msg->packet.size());
    msg->packet.push_back(static_cast<uint8_t>(crc >> 8));    // CRC16 high byte
    msg->packet.push_back(static_cast<uint8_t>(crc & 0xFF));  // CRC16 low byte

    assert(msg->VerifyPacketSerialization() && ("Packet serialization verification failed for CommandID: " + std::to_string(static_cast<int>(id))).c_str());
    
    return msg;
}

std::shared_ptr<Message> CreateSuccessAckMessage(CommandID::Type id)
{
    auto msg = std::make_shared<Message>(id, 1);
    msg->AddToPacket(ACK_SUCCESS);

    // ACKs are 6 bytes: 5 header + 1 status (NO CRC, NO PADDING)
    
    return msg;
}


std::shared_ptr<Message> CreateErrorAckMessage(CommandID::Type id, uint8_t error_code)
{
    auto msg = std::make_shared<Message>(id, 1);  // data_length = 1 (just the error code)
    msg->AddToPacket(error_code);  // Error code is the status byte

    // NACKs are 6 bytes: 5 header + 1 error code (NO CRC, NO PADDING)
    
    return msg;
}


void SerializeToBytes(uint64_t value, std::vector<uint8_t>& output)
{
    output.push_back(static_cast<uint8_t>(value >> 56));
    output.push_back(static_cast<uint8_t>(value >> 48));
    output.push_back(static_cast<uint8_t>(value >> 40));
    output.push_back(static_cast<uint8_t>(value >> 32));
    output.push_back(static_cast<uint8_t>(value >> 24));
    output.push_back(static_cast<uint8_t>(value >> 16));
    output.push_back(static_cast<uint8_t>(value >> 8));
    output.push_back(static_cast<uint8_t>(value & 0xFF));
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

// CRC16-CCITT implementation
uint16_t calculate_crc16(const uint8_t* data, size_t length)
{
    uint16_t crc = 0xFFFF; 
    const uint16_t polynomial = 0x1021; 
    
    for (size_t i = 0; i < length; i++)
    {
        crc ^= (static_cast<uint16_t>(data[i]) << 8); 
        
        for (uint8_t bit = 0; bit < 8; bit++)
        {
            if (crc & 0x8000)  
            {
                crc = (crc << 1) ^ polynomial;
            }
            else
            {
                crc = crc << 1;
            }
        }
    }
    
    return crc;
}
