#include "payload.hpp"
#include <iostream>

const char* ToString(PayloadState state) {
    switch (state) {
        case PayloadState::STARTUP: return "STARTUP";
        case PayloadState::NOMINAL: return "NOMINAL";
        case PayloadState::IDLE: return "IDLE";
        case PayloadState::SAFE_MODE: return "SAFE_MODE";
        default: return "UNKNOWN";
    }
}



Payload::Payload()
:
state(PayloadState::STARTUP)
{
    // Constructor
}


void Payload::run()
{
    // Run the payload
    std::cout << "Payload is running" << std::endl;
}