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
state(PayloadState::STARTUP),
camera(Camera(0))
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

    // For now - to nominal
    SwitchToState(PayloadState::NOMINAL);
    // SPDLOG_INFO("Payload state is: {}", ToString(state));
}

void Payload::AddCommand(int command_id, std::vector<uint8_t>& data, int priority)
{
    size_t cmd_id = static_cast<size_t>(command_id);


    // Look up the corresponding function with the command ID
    if (cmd_id < COMMAND_NUMBER) 
    {
        CommandFunction cmd_function = COMMAND_FUNCTIONS[cmd_id];

        // Create task object 
        Task task(command_id, cmd_function, data, this, priority, COMMAND_NAMES[cmd_id]);

        // Add task to the RX queue
        rx_queue.AddTask(task);
    } 
    else 
    {
        // TODO: Handle invalid command ID case
        // Will transmit error message through comms
        SPDLOG_WARN("Invalid command ID");
    }

}

void Payload::Run()
{   

    SPDLOG_INFO("Starting Payload Task Manager");

    // Launch camera system
    StartCameraThread();

    // Launch communication system
    // TODO 

    // Running execution loop 

    while (true) 
    {
        // Check for incoming commands
        if (!rx_queue.IsEmpty()) 
        {
            Task task = std::move(rx_queue.GetNextTask());
            task.Execute();
        }

        // Check for outgoing commands
        /*if (!tx_queue.IsEmpty()) 
        {
            Task task = tx_queue.GetNextTask();
            task.Execute();
        }*/

        // Sleep for a while
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // rx_queue.PrintAllTasks();
    }
    
}

const RX_Queue& Payload::GetRxQueue() const {
    return rx_queue;
}

RX_Queue& Payload::GetRxQueue() {
    return rx_queue;
}


const TX_Queue& Payload::GetTxQueue() const {
    return tx_queue;
}


TX_Queue& Payload::GetTxQueue() {
    return tx_queue;
}


const Camera& Payload::GetCamera() const {
    return camera;
}

Camera& Payload::GetCamera() {
    return camera;
}




const PayloadState& Payload::GetState() const {
    return state;
}

void Payload::StartCameraThread()
{
    // Launch camera thread
    camera_thread = std::thread(&Camera::RunLoop, &camera);
}


void Payload::StopCameraThread()
{
    camera.StopLoop();
    camera_thread.join();
}