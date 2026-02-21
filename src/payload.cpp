#include <filesystem>
#include "payload.hpp"
#include "core/data_handling.hpp"
#include "core/timing.hpp"
#include "image_sender.hpp"


const char* ToString(PayloadState state) {
    switch (state) {
        case PayloadState::STARTUP: return "STARTUP";
        case PayloadState::NOMINAL: return "NOMINAL";
        default: return "UNKNOWN";
    }
}

// Singleton instance creation
Payload& Payload::CreateInstance(std::unique_ptr<Configuration> config, std::unique_ptr<Communication> comms_interface)
{
    static Payload instance(std::move(config), std::move(comms_interface));
    return instance;
}

// Accessor for the existing instance after initialization
Payload& Payload::GetInstance()
{
    return CreateInstance(nullptr, nullptr);
}



Payload::Payload(std::unique_ptr<Configuration> _config, std::unique_ptr<Communication> _comms_interface)
:
_running_instance(false),
config(std::move(_config)),
communication(std::move(_comms_interface)),
camera_manager(config->GetCameraConfigs()),
state(PayloadState::STARTUP),
thread_pool(std::make_unique<ThreadPool>(5))
{   

    timing::InitializeBootTime(); // Just in case
    SPDLOG_INFO("Configuration read successfully");
    SPDLOG_INFO("Payload state initialized to: {}", ToString(state)); 
    
}

Payload::~Payload()
{
    Stop();
    StopCommunicationThread();
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
        Task task(cmd_id, cmd_function, data, priority, COMMAND_NAMES[cmd_id]);

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


void Payload::Run()
{   

    SPDLOG_INFO("Starting Payload Task Manager");

    // Launch communication system
    StartCommunicationThread();

    // Launch camera system
    StartCameraThread();

    // Launch OD system
    StartODThread();

    // Launch telemetry service
    StartTelemetryService();

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
    }
    
    SPDLOG_INFO("Exiting Payload Run Loop");
}

void Payload::Stop()
{
    // Stop execution loop
    _running_instance = false;

    // Stop the telemetry service
    StopTelemetryService();

    // Stop OD system
    StopODThread();

    // Stop camera system
    StopCameraThread();

    // Stop thread pool
    StopThreadPool();

    // Stop communication system
    // StopCommunicationThread(); ~ done in destructor so we can ACK the command

    
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
    std::array<bool, NUM_CAMERAS> temp;
    [[maybe_unused]] int nb_enabled_cams = camera_manager.EnableCameras(temp);
    camera_thread = std::thread(&CameraManager::RunLoop, &camera_manager);
}


void Payload::StopCameraThread()
{
    camera_manager.StopLoops();
    if (camera_thread.joinable())
    {
        camera_thread.join();
    }
}

void Payload::StopThreadPool()
{
    if (thread_pool) 
    {
        thread_pool->shutdown();
        thread_pool.reset(nullptr);
    }
    else 
    {
        SPDLOG_WARN("ThreadPool is nullptr in StopThreadPool.");
    }
}

void Payload::StartCommunicationThread()
{
    communication->Connect();
    communication_thread = std::thread(&Communication::RunLoop, communication.get());
    SPDLOG_INFO("Communication thread started");
}

void Payload::StopCommunicationThread()
{
    communication->StopLoop();
    if (communication_thread.joinable())
    {
        SPDLOG_INFO("Joining communication thread...");
        communication_thread.join();
    }
    SPDLOG_INFO("Communication thread stopped");
    communication->Disconnect();
}

void Payload::StartTelemetryService()
{
    telemetry_thread = std::thread(&Telemetry::RunService, &telemetry);
    SPDLOG_INFO("Telemetry thread started");

}


void Payload::StopTelemetryService()
{
    telemetry.StopService();
    if (telemetry_thread.joinable())
    {
        SPDLOG_INFO("Joining telemetry thread...");
        telemetry_thread.join();
    }
    SPDLOG_INFO("Telemetry thread stopped");
    
}

const Telemetry& Payload::GetTelemetry() const
{
    return telemetry;
}

Telemetry& Payload::GetTelemetry()
{
    return telemetry;
}

void Payload::StartODThread()
{
    // Launch OD thread
    od_thread = std::thread(&OD::RunLoop, &od);
    SPDLOG_INFO("OD thread started");
}

void Payload::StopODThread()
{
    od.StopLoop();
    if (od_thread.joinable())
    {
        SPDLOG_INFO("Joining OD thread...");
        od_thread.join();
    }
    SPDLOG_INFO("OD thread stopped");
}

const OD& Payload::GetOD() const
{
    return od;
}

OD& Payload::GetOD()
{
    return od;
}

size_t Payload::GetNbTasksInExecution()
{
    if (thread_pool)
    {
        return thread_pool->GetBusyThreadCount();
    }
    else
    {
        return 0;
    }
}

void Payload::SetLastExecutedCmdID(uint8_t cmd_id)
{
    last_executed_cmd_id.store(cmd_id);
    last_executed_cmd_time.store(timing::GetUptimeMs());
}

uint8_t Payload::GetLastExecutedCmdID() const
{
    return last_executed_cmd_id.load();
}

uint32_t Payload::GetLastExecutedCmdTime() const
{
    return last_executed_cmd_time.load();
}
