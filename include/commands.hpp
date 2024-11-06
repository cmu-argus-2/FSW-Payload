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
typedef std::function<void(Payload&, std::vector<uint8_t>&)> CommandFunction;

// Command IDs within a namespace
// Chose this instead of enum class so command IDs can be accessed without casting
namespace CommandID {
    enum Type : uint8_t {
        NEUTRAL_ACK = 0,
        REQUEST_STATE = 1,
        SHUTDOWN = 2,
        SYNCHRONIZE_TIME = 3,
        RUN_SELF_DIAGNOSTICS = 4,
        REQUEST_LAST_TELEMETRY = 5,
        SET_TELEMETRY_FREQUENCY = 6,
        TURN_ON_CAMERAS = 7,
        TURN_OFF_CAMERAS = 8,
        ENABLE_CAMERA_X = 9,
        DISABLE_CAMERA_X = 10,
        CAPTURE_IMAGES = 11,
        START_CAPTURE_IMAGES_EVERY_X_SECONDS = 12,
        STOP_CAPTURE_IMAGES = 13,
        STORED_IMAGES = 14,
        REQUEST_LAST_IMAGE = 15,
        IMG_TRANSFER_COMPLETE_ACK = 16,
        DELETE_ALL_STORED_IMAGES = 17,
        ENABLE_REGION_X = 18,
        DISABLE_REGION_X = 19,
        RUN_OD = 20,
        DEBUG_DISPLAY_CAMERA = 21
    };
}

// Array of all Command IDs
constexpr CommandID::Type ALL_COMMAND_IDS[] = 
{
    CommandID::NEUTRAL_ACK,
    CommandID::REQUEST_STATE,
    CommandID::SHUTDOWN,
    CommandID::SYNCHRONIZE_TIME,
    CommandID::RUN_SELF_DIAGNOSTICS,
    CommandID::REQUEST_LAST_TELEMETRY,
    CommandID::SET_TELEMETRY_FREQUENCY,
    CommandID::TURN_ON_CAMERAS,
    CommandID::TURN_OFF_CAMERAS,
    CommandID::ENABLE_CAMERA_X,
    CommandID::DISABLE_CAMERA_X,
    CommandID::CAPTURE_IMAGES,
    CommandID::START_CAPTURE_IMAGES_EVERY_X_SECONDS,
    CommandID::STOP_CAPTURE_IMAGES,
    CommandID::STORED_IMAGES,
    CommandID::REQUEST_LAST_IMAGE,
    CommandID::IMG_TRANSFER_COMPLETE_ACK,
    CommandID::DELETE_ALL_STORED_IMAGES,
    CommandID::ENABLE_REGION_X,
    CommandID::DISABLE_REGION_X,
    CommandID::RUN_OD,
    CommandID::DEBUG_DISPLAY_CAMERA
};

constexpr uint8_t COMMAND_NUMBER = sizeof(ALL_COMMAND_IDS) / sizeof(ALL_COMMAND_IDS[0]);

// Command functions declarations
void neutral_ack(Payload& payload, std::vector<uint8_t>& data);
void request_state(Payload& payload, std::vector<uint8_t>& data);
void shutdown(Payload& payload, std::vector<uint8_t>& data);
void synchronize_time(Payload& payload, std::vector<uint8_t>& data);
void run_self_diagnostics(Payload& payload, std::vector<uint8_t>& data);
void request_last_telemetry(Payload& payload, std::vector<uint8_t>& data);
void set_telemetry_frequency(Payload& payload, std::vector<uint8_t>& data);
void turn_on_cameras(Payload& payload, std::vector<uint8_t>& data);
void turn_off_cameras(Payload& payload, std::vector<uint8_t>& data);
void enable_camera_x(Payload& payload, std::vector<uint8_t>& data);
void disable_camera_x(Payload& payload, std::vector<uint8_t>& data);
void capture_images(Payload& payload, std::vector<uint8_t>& data);
void start_capture_images_every_x_seconds(Payload& payload, std::vector<uint8_t>& data);
void stop_capture_images(Payload& payload, std::vector<uint8_t>& data);
void stored_images(Payload& payload, std::vector<uint8_t>& data);
void request_last_image(Payload& payload, std::vector<uint8_t>& data);
void img_transfer_complete_ack(Payload& payload, std::vector<uint8_t>& data);
void delete_all_stored_images(Payload& payload, std::vector<uint8_t>& data);
void enable_region_x(Payload& payload, std::vector<uint8_t>& data);
void disable_region_x(Payload& payload, std::vector<uint8_t>& data);
void run_od(Payload& payload, std::vector<uint8_t>& data);
void debug_display_camera(Payload& payload, std::vector<uint8_t>& data);



// Array mapping CommandID to corresponding functions
extern std::array<CommandFunction, COMMAND_NUMBER> COMMAND_FUNCTIONS;

// Array mapping CommandID to corresponding names (for easier debugging)
extern std::array<std::string_view, COMMAND_NUMBER> COMMAND_NAMES;

#endif // COMMANDS_HPP
