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
    std::cout << "Payload state initialized to: " << ToString(state) << std::endl; // TODO Logging
}

void Payload::SwitchToState(PayloadState new_state) 
{
    state = new_state;
    std::cout << "Payload state changed to: " << ToString(state) << std::endl; // TODO Logging
}



void Payload::Initialize()
{
    // Run startup health procedures
    RunStartupHealthProcedures();
    // Retrieve internal states
    RetrieveInternalStates();

    // Switch to nominal state
    SwitchToState(PayloadState::NOMINAL);
}

void Payload::RunStartupHealthProcedures()
{
    // Run startup health procedures
    std::cout << "Running startup health procedures" << std::endl;
}


void Payload::RetrieveInternalStates()
{
    // Retrieve internal states
    std::cout << "Retrieving internal states" << std::endl;
}


void Payload::run()
{   

    Initialize();

    // Run the payload
    std::cout << "Payload is running" << std::endl;
}


const RX_Queue& Payload::GetRxQueue() const {
    return rx_queue;
}


const TX_Queue& Payload::GetTxQueue() const {
    return tx_queue;
}