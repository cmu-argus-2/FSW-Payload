#include "commands.hpp"
#include "payload.hpp"


// Command functions array definition 
std::array<CommandFunction, COMMAND_NUMBER> COMMAND_FUNCTIONS = 
{
    start,
    shutdown,
    request_state
};



void start(Payload* payload, std::vector<uint8_t> data)
{
    SPDLOG_INFO("Payload start");
    // TODO
}


void shutdown(Payload* payload, std::vector<uint8_t> data)
{
    SPDLOG_INFO("Payload shutdown");
    // TODO
}


void request_state(Payload* payload, std::vector<uint8_t> data)
{
    payload->GetState();
    SPDLOG_INFO("State is: {} ", ToString(payload->GetState()));
}