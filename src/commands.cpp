#include "commands.hpp"
#include "payload.hpp"


// Command functions array definition 
std::array<CommandFunction, COMMAND_NUMBER> COMMAND_FUNCTIONS = 
{
    start,
    shutdown,
    request_state,
    display_camera
};

// Define the array of strings mapping CommandID to command names
std::array<std::string, COMMAND_NUMBER> COMMAND_NAMES = {
    "START",
    "SHUTDOWN",
    "REQUEST_STATE",
    "DISPLAY_CAMERA"
};




void start(Payload* payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Payload start");
    // TODO
}


void shutdown(Payload* payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Initiating Payload shutdown..");
    payload->Stop();
}


void request_state(Payload* payload, std::vector<uint8_t>& data)
{
    payload->GetState();
    SPDLOG_INFO("State is: {} ", ToString(payload->GetState()));
}


void display_camera(Payload* payload, std::vector<uint8_t>& data)
{
    SPDLOG_INFO("Activating the display of the camera");
    payload->GetCamera().DisplayLoop(true);
}