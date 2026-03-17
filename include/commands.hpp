#ifndef COMMANDS_HPP
#define COMMANDS_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <string>
#include <functional>

#define PING_RESP_VALUE 0x60 // DO NOT CHANGE THIS VALUE

// Command function type
typedef std::function<void(std::vector<uint8_t>&)> CommandFunction;

// Command IDs within a namespace
// Chose this instead of enum class so command IDs can be accessed without casting
// Could certainly be better but not worth changing at this point
namespace CommandID {
    enum Type : uint8_t 
    {
        PING_ACK = 0,
        SHUTDOWN = 1,
        REQUEST_TELEMETRY = 2,
        ENABLE_CAMERAS = 3,
        DISABLE_CAMERAS = 4,
        CAPTURE_IMAGES = 5,
        START_CAPTURE_IMAGES_PERIODICALLY = 6,
        STOP_CAPTURE_IMAGES = 7,
        REQUEST_STORAGE_INFO = 8,
        REQUEST_IMAGE = 9,
        REQUEST_NEXT_FILE_PACKET = 10,
        CLEAR_STORAGE = 11,
        PING_OD_STATUS = 12,
        RUN_OD = 13,
        REQUEST_OD_RESULT = 14,
        SYNCHRONIZE_TIME = 15,
        FULL_RESET = 16,
        DEBUG_DISPLAY_CAMERA = 17,
        DEBUG_STOP_DISPLAY = 18,
        REQUEST_NEXT_FILE_PACKETS = 19, 
        START_ROI_CAPTURE = 20
    };
}

// Array of all Command IDs
inline constexpr CommandID::Type ALL_COMMAND_IDS[] = 
{
    CommandID::PING_ACK,
    CommandID::SHUTDOWN,
    CommandID::REQUEST_TELEMETRY,
    CommandID::ENABLE_CAMERAS,
    CommandID::DISABLE_CAMERAS,
    CommandID::CAPTURE_IMAGES,
    CommandID::START_CAPTURE_IMAGES_PERIODICALLY,
    CommandID::STOP_CAPTURE_IMAGES,
    CommandID::REQUEST_STORAGE_INFO,
    CommandID::REQUEST_IMAGE,
    CommandID::REQUEST_NEXT_FILE_PACKET,
    CommandID::CLEAR_STORAGE,
    CommandID::PING_OD_STATUS,
    CommandID::RUN_OD,
    CommandID::REQUEST_OD_RESULT,
    CommandID::SYNCHRONIZE_TIME,
    CommandID::FULL_RESET,
    CommandID::DEBUG_DISPLAY_CAMERA,
    CommandID::DEBUG_STOP_DISPLAY,
    CommandID::REQUEST_NEXT_FILE_PACKETS,
    CommandID::START_ROI_CAPTURE
};

constexpr uint8_t COMMAND_NUMBER = sizeof(ALL_COMMAND_IDS) / sizeof(ALL_COMMAND_IDS[0]);

// Command functions declarations
void ping_ack(std::vector<uint8_t>& data);
void shutdown(std::vector<uint8_t>& data);
void request_telemetry(std::vector<uint8_t>& data);
void enable_cameras(std::vector<uint8_t>& data);
void disable_cameras(std::vector<uint8_t>& data);
void capture_images(std::vector<uint8_t>& data);
void start_capture_images_periodically(std::vector<uint8_t>& data);
void stop_capture_images(std::vector<uint8_t>& data);
void request_storage_info(std::vector<uint8_t>& data);
void request_image(std::vector<uint8_t>& data);
void request_next_file_packet(std::vector<uint8_t>& data);
void clear_storage(std::vector<uint8_t>& data);
void ping_od_status(std::vector<uint8_t>& data);
void run_od(std::vector<uint8_t>& data);
void request_od_result(std::vector<uint8_t>& data);
void synchronize_time(std::vector<uint8_t>& data);
void full_reset(std::vector<uint8_t>& data);
void debug_display_camera(std::vector<uint8_t>& data);
void debug_stop_display(std::vector<uint8_t>& data);
void request_next_file_packets(std::vector<uint8_t>& data);
void start_roi_capture(std::vector<uint8_t>& data);


// Array mapping CommandID to corresponding functions
extern std::array<CommandFunction, COMMAND_NUMBER> COMMAND_FUNCTIONS;

// Array mapping CommandID to corresponding names (for easier debugging)
extern std::array<std::string_view, COMMAND_NUMBER> COMMAND_NAMES;

#endif // COMMANDS_HPP
