#ifndef COMMANDS_HPP
#define COMMANDS_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <string>
#include <functional>

// Forward declaration of Payload class
class Payload;

// Command function type
typedef std::function<void(Payload*, std::vector<uint8_t>&)> CommandFunction;

// Command IDs within a namespace
// Chose this instead of enum class so command IDs can be accessed without casting
namespace CommandID {
    enum Type : uint8_t {
        START = 0x00,
        SHUTDOWN = 0x01,
        REQUEST_STATE = 0x02,
        DISPLAY_CAMERA = 0x03
    };
}

// Array of all Command IDs
constexpr CommandID::Type ALL_COMMAND_IDS[] = 
{
    CommandID::START,
    CommandID::SHUTDOWN,
    CommandID::REQUEST_STATE,
    CommandID::DISPLAY_CAMERA
};

constexpr uint8_t COMMAND_NUMBER = sizeof(ALL_COMMAND_IDS) / sizeof(ALL_COMMAND_IDS[0]);

// Command functions declarations
void start(Payload* payload, std::vector<uint8_t>& data);
void shutdown(Payload* payload, std::vector<uint8_t>& data);
void request_state(Payload* payload, std::vector<uint8_t>& data);
void display_camera(Payload* payload, std::vector<uint8_t>& data);

// Array mapping CommandID to corresponding functions
extern std::array<CommandFunction, COMMAND_NUMBER> COMMAND_FUNCTIONS;

// Array mapping CommandID to corresponding names (for easier debugging)
extern std::array<std::string, COMMAND_NUMBER> COMMAND_NAMES;

#endif // COMMANDS_HPP
