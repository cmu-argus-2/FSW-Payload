#include "payload.hpp"

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
    SPDLOG_INFO("Payload state initialized to: {}", ToString(state)); 
}

void Payload::SwitchToState(PayloadState new_state) 
{
    state = new_state;
    SPDLOG_INFO("Payload state changed to: {}", ToString(state)); 
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
    SPDLOG_INFO("Running startup health procedures");
    // TODO 
}


void Payload::RetrieveInternalStates()
{
    // Retrieve internal states
    SPDLOG_INFO("Retrieving internal states");
    // TODO
}

void Payload::AddCommandToQueue(CommandID command_id, const std::vector<uint8_t>& data)
{
    size_t cmd_id = static_cast<size_t>(command_id);

    // Look up the corresponding function with the command ID
    if (cmd_id < COMMAND_NUMBER) 
    {
        CommandFunction cmd_function = COMMAND_FUNCTIONS[cmd_id];

        // Create task object 
        Task task(int(cmd_id), cmd_function, data, this, 0);

        // Add task to the RX queue
        rx_queue.AddTask(task);
        SPDLOG_INFO("Command added to RX queue"); 
    } 
    else 
    {
        // TODO: Handle invalid command ID case
        // Will transmit error message through comms
        SPDLOG_INFO("Invalid command ID");
    }






}

void Payload::Run()
{   

    Initialize();

    // Run the payload
    SPDLOG_INFO("Payload is running");
}


const RX_Queue& Payload::GetRxQueue() const {
    return rx_queue;
}


const TX_Queue& Payload::GetTxQueue() const {
    return tx_queue;
}