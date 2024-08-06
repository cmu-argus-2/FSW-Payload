#include "commands.hpp"
#include "payload.hpp"
#include <iostream>

// Command functions array definition 
std::array<CommandFunction, COMMAND_NUMBER> COMMAND_FUNCTIONS = 
{
    start,
    shutdown,
    request_state
};



void start(Payload* payload, std::vector<uint8_t> data)
{
    std::cout << "Payload start" << std::endl;
    // TODO
}


void shutdown(Payload* payload, std::vector<uint8_t> data)
{
    std::cout << "Payload shutdown" << std::endl;
    // TODO
}


void request_state(Payload* payload, std::vector<uint8_t> data)
{
    payload->GetState();
    std::cout << "State is: " << ToString(payload->GetState()) << std::endl;
}