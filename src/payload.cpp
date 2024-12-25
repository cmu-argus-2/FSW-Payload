#include <filesystem>
#include "payload.hpp"
#include "core/data_handling.hpp"

const char* ToString(PayloadState state) {
    switch (state) {
        case PayloadState::STARTUP: return "STARTUP";
        case PayloadState::NOMINAL: return "NOMINAL";
        case PayloadState::IDLE: return "IDLE";
        case PayloadState::SAFE_MODE: return "SAFE_MODE";
        default: return "UNKNOWN";
    }
}

Payload& Payload::GetInstance(Configuration& config, std::unique_ptr<Communication> comms_interface)
{
    static Payload instance(config, std::move(comms_interface));
    return instance;
}


Payload::Payload(Configuration& config, std::unique_ptr<Communication> comms_interface)
:
_running_instance(false),
config(config),
communication(std::move(comms_interface)),
camera_manager(config.GetCameraConfigs()),
state(PayloadState::STARTUP),
thread_pool(std::make_unique<ThreadPool>(4))
{   

    SPDLOG_INFO("Configuration read successfully");
    SPDLOG_INFO("Payload state initialized to: {}", ToString(state)); 
    
}


void Payload::ReadNewConfiguration(Configuration& config)
{
    // Read New Configuration
    this->config = config;
    SPDLOG_INFO("New reconfiguration read successfully");
    // TODO

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


    // Initialize data storage
    DH::InitializeDataStorage();


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

void Payload::AddCommand(uint8_t cmd_id, std::vector<uint8_t>& data, uint8_t priority)
{
    // Look up the corresponding function with the command ID
    if (cmd_id < COMMAND_NUMBER) 
    {
        CommandFunction cmd_function = COMMAND_FUNCTIONS[cmd_id];

        // Create task object 
        Task task(cmd_id, cmd_function, data, this, priority, COMMAND_NAMES[cmd_id]);

        // Add task to the RX queue
        rx_queue.AddTask(task);
    } 
    else 
    {
        // TODO: Handle invalid command ID case
        // Will transmit error message through comms
        SPDLOG_WARN("Invalid command ID");
    }

    cv_queue.notify_one();

}

void Payload::TransmitMessage(std::shared_ptr<Message> msg)
{
    // TODO check if the message id is valid
    
    // Add message to the TX queue
    tx_queue.AddMsg(msg);
}


Configuration& Payload::GetConfiguration()
{
    return config;
}

void Payload::Run()
{   

    SPDLOG_INFO("Starting Payload Task Manager");

    // Launch camera system
    StartCameraThread();

    // Launch communication system
    _StartCommunicationThread();

    // Running execution loop 
    _running_instance = true;



    while (_running_instance) 
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv_queue.wait(lock, [this] { return !_running_instance || !rx_queue.IsEmpty(); });

        if (!_running_instance) {
            break;
        }
            
        // Check for incoming commands
        if (!rx_queue.IsEmpty() && thread_pool) {
            Task task = std::move(rx_queue.GetNextTask());


            if (task.GetID() == CommandID::SHUTDOWN) 
            {
                task.Execute(); // Execute the shutdown task
                break;
            }

            thread_pool->enqueue([task]() mutable { task.Execute(); }); // Capturing `task` by value
        }

        // Check for outgoing messages - Comms thread responsible for this
        /*if (!tx_queue.IsEmpty()) 
        {
            std::shared_ptr<Message> msg = tx_queue.GetNextMsg();
        }*/
    }
    
    SPDLOG_INFO("Exiting Payload Run Loop");
}

void Payload::Stop()
{
    // Stop execution loop
    _running_instance = false;

    // Stop camera system
    StopCameraThread();

    // Stop thread pool
    StopThreadPool();

    // Stop communication system
    _StopCommunicationThread();

    
    SPDLOG_WARN("Payload Shutdown");
}

bool Payload::IsRunning() const
{
    return _running_instance;
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


const CameraManager& Payload::GetCameraManager() const {
    return camera_manager;
}

CameraManager& Payload::GetCameraManager() {
    return camera_manager;
}

const PayloadState& Payload::GetState() const {
    return state;
}

void Payload::StartCameraThread()
{
    // Launch camera thread
    camera_manager.TurnOn();
    camera_thread = std::thread(&CameraManager::RunLoop, &camera_manager, this);
}


void Payload::StopCameraThread()
{
    camera_manager.StopLoops();
    camera_thread.join();
}

void Payload::StopThreadPool()
{

    thread_pool->shutdown();
    thread_pool.reset(nullptr);

}

void Payload::_StartCommunicationThread()
{
    
    communication->Connect();
    communication_thread = std::thread(&Communication::RunLoop, communication.get(), this);
    SPDLOG_INFO("Communication thread started");
}

void Payload::_StopCommunicationThread()
{
    communication->StopLoop();
    communication_thread.join();
    SPDLOG_INFO("Communication thread stopped");
    communication->Disconnect();
}