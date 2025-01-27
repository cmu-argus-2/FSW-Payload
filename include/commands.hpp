#ifndef COMMANDS_HPP
#define COMMANDS_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <string>
#include <functional>

#define PING_VALUE 0x60

// Forward declaration of Payload class
class Payload;

// Command function type
typedef std::function<void(Payload&, std::vector<uint8_t>&)> CommandFunction;

// Command IDs within a namespace
// Chose this instead of enum class so command IDs can be accessed without casting
namespace CommandID {
    enum Type : uint8_t 
    {
        PING_ACK = 0,
        SHUTDOWN = 1,
        SYNCHRONIZE_TIME = 2,
        REQUEST_TELEMETRY = 3,
        ENABLE_CAMERAS = 4,
        DISABLE_CAMERAS = 5,
        CAPTURE_IMAGES = 6,
        START_CAPTURE_IMAGES_PERIODICALLY = 7,
        STOP_CAPTURE_IMAGES = 8,
        STORED_IMAGES = 9,
        REQUEST_IMAGE = 10,
        DELETE_IMAGES = 11,
        RUN_OD = 12,
        PING_OD_STATUS = 13,
        DEBUG_DISPLAY_CAMERA = 14,
        DEBUG_STOP_DISPLAY = 15
    };
}

// Array of all Command IDs
constexpr CommandID::Type ALL_COMMAND_IDS[] = 
{
    CommandID::PING_ACK,
    CommandID::SHUTDOWN,
    CommandID::SYNCHRONIZE_TIME,
    CommandID::REQUEST_TELEMETRY,
    CommandID::ENABLE_CAMERAS,
    CommandID::DISABLE_CAMERAS,
    CommandID::CAPTURE_IMAGES,
    CommandID::START_CAPTURE_IMAGES_PERIODICALLY,
    CommandID::STOP_CAPTURE_IMAGES,
    CommandID::STORED_IMAGES,
    CommandID::REQUEST_IMAGE,
    CommandID::DELETE_IMAGES,
    CommandID::RUN_OD,
    CommandID::PING_OD_STATUS,
    CommandID::DEBUG_DISPLAY_CAMERA,
    CommandID::DEBUG_STOP_DISPLAY
};

constexpr uint8_t COMMAND_NUMBER = sizeof(ALL_COMMAND_IDS) / sizeof(ALL_COMMAND_IDS[0]);

// Command functions declarations
void ping_ack(Payload& payload, std::vector<uint8_t>& data);
void shutdown(Payload& payload, std::vector<uint8_t>& data);
void synchronize_time(Payload& payload, std::vector<uint8_t>& data);
void request_telemetry(Payload& payload, std::vector<uint8_t>& data);
void enable_cameras(Payload& payload, std::vector<uint8_t>& data);
void disable_cameras(Payload& payload, std::vector<uint8_t>& data);
void capture_images(Payload& payload, std::vector<uint8_t>& data);
void start_capture_images_periodically(Payload& payload, std::vector<uint8_t>& data);
void stop_capture_images(Payload& payload, std::vector<uint8_t>& data);
void stored_images(Payload& payload, std::vector<uint8_t>& data);
void request_image(Payload& payload, std::vector<uint8_t>& data);
void delete_images(Payload& payload, std::vector<uint8_t>& data);
void run_od(Payload& payload, std::vector<uint8_t>& data);
void ping_od_status(Payload& payload, std::vector<uint8_t>& data);
void debug_display_camera(Payload& payload, std::vector<uint8_t>& data);
void debug_stop_display(Payload& payload, std::vector<uint8_t>& data);


// Array mapping CommandID to corresponding functions
extern std::array<CommandFunction, COMMAND_NUMBER> COMMAND_FUNCTIONS;

// Array mapping CommandID to corresponding names (for easier debugging)
extern std::array<std::string_view, COMMAND_NUMBER> COMMAND_NAMES;

#endif // COMMANDS_HPP
