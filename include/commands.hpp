#ifndef COMMANDS_HPP
#define COMMANDS_HPP

#include <cstdint>
#include <cstddef>
#include <vector>

// Forward declaration of Payload class
class Payload;


// Command IDs

enum class CommandID : uint8_t {
    START = 0x10,
    SHUTDOWN = 0x11,
    REQUEST_STATE = 0x12
};

constexpr CommandID ALL_COMMAND_IDS[] = 
{
    CommandID::START,
    CommandID::SHUTDOWN,
    CommandID::REQUEST_STATE
};


constexpr size_t COMMAND_NUMBER = sizeof(ALL_COMMAND_IDS) / sizeof(ALL_COMMAND_IDS[0]);


// Command functions declarations
void start(Payload* payload, std::vector<uint8_t> data);
void shutdown(Payload* payload, std::vector<uint8_t> data);
void request_state(Payload* payload, std::vector<uint8_t> data);











#endif // COMMANDS_HPP