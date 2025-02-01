#ifndef PAYLOAD_HPP
#define PAYLOAD_HPP

#include <thread>
#include <atomic>
#include <condition_variable>
#include <memory>
#include "spdlog/spdlog.h"

#include "configuration.hpp"
#include "queues.hpp"
#include "commands.hpp"
#include "core/thread_pool.hpp"
#include "vision/camera_manager.hpp"
#include "navigation/od.hpp"
#include "communication/named_pipe.hpp"
#include "communication/uart.hpp"
#include "telemetry/telemetry.hpp"

enum class PayloadState : uint8_t {
    STARTUP = 0x00,
    NOMINAL = 0x01
};

// Function to convert PayloadState enum to string
const char* ToString(PayloadState state);

class Payload
{
public:

    // First-time initialization; subsequent calls return the same instance.
    static Payload& CreateInstance(std::unique_ptr<Configuration> _config, std::unique_ptr<Communication> _comms_interface);
    // Accessor for the existing instance (after initialization).
    static Payload& GetInstance();

    void Initialize();
    const PayloadState& GetState() const;

    void Run();
    void Stop();
    bool IsRunning() const;

    void AddCommand(uint8_t command_id, std::vector<uint8_t>& data, uint8_t priority = 0);
    void TransmitMessage(std::shared_ptr<Message> msg);

    const RX_Queue& GetRxQueue() const;
    RX_Queue& GetRxQueue(); 
    const TX_Queue& GetTxQueue() const; 
    TX_Queue& GetTxQueue(); 

    const CameraManager& GetCameraManager() const; 
    CameraManager& GetCameraManager(); 

    const Telemetry& GetTelemetry() const;
    Telemetry& GetTelemetry();

    const OD& GetOD() const;
    OD& GetOD();

    void SetLastExecutedCmdID(uint8_t cmd_id);
    uint8_t GetLastExecutedCmdID() const;

private:

    Payload(std::unique_ptr<Configuration> config, std::unique_ptr<Communication> comms_interface);
    ~Payload();

    // Singleton constraints
    Payload(const Payload&) = delete;
    void operator=(const Payload&) = delete;

    // static std::unique_ptr<Payload> _instance; // Singleton instance

    std::atomic<bool> _running_instance;

    std::unique_ptr<Configuration> config;
    PayloadState state;
    RX_Queue rx_queue;
    TX_Queue tx_queue;

    std::mutex mtx;
    std::condition_variable cv_queue;

    void SwitchToState(PayloadState new_state);
    void RunStartupHealthProcedures();
    void RetrieveInternalStates();

    // Communication interface
    std::unique_ptr<Communication> communication;
    std::thread communication_thread;
    void StartCommunicationThread();
    void StopCommunicationThread();

    // Camera interface
    CameraManager camera_manager;
    std::thread camera_thread;
    void StartCameraThread();
    void StopCameraThread();

    // OD interface
    OD od;
    std::thread od_thread;
    void StartODThread();
    void StopODThread();

    // Thread Pool
    std::unique_ptr<ThreadPool> thread_pool;
    void StopThreadPool();
    std::atomic<uint8_t> last_executed_cmd_id = 99; // None


    // Telemetry
    std::thread telemetry_thread;
    Telemetry telemetry;
    void StartTelemetryService();
    void StopTelemetryService();

};

// Inline helper functions to access the payload instance throughout the codebase
static inline Payload& payload() { return Payload::GetInstance(); }
static inline CameraManager& cameraManager() { return payload().GetCameraManager(); }
static inline OD& od() { return payload().GetOD(); }
static inline Telemetry& telemetry() { return payload().GetTelemetry(); }

#endif // PAYLOAD_HPP
