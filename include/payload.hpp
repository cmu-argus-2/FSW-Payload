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
#include "core/data_handling.hpp"
#include "core/thread_pool.hpp"
#include "vision/camera_manager.hpp"


enum class PayloadState : uint8_t {
    STARTUP = 0x00,
    IDLE = 0x01,
    NOMINAL = 0x02,
    SAFE_MODE = 0x03
};

// Function to convert PayloadState enum to string
const char* ToString(PayloadState state);

class Payload
{
public:
    Payload(Configuration& config);
    // ~Payload();

    void Initialize();
    const PayloadState& GetState() const;

    void Run();
    void Stop();

    void AddCommand(uint8_t command_id, std::vector<uint8_t>& data, uint8_t priority = 0);
    void TransmitMessage(std::shared_ptr<Message> msg);

    const RX_Queue& GetRxQueue() const;
    RX_Queue& GetRxQueue(); 
    const TX_Queue& GetTxQueue() const; 
    TX_Queue& GetTxQueue(); 

    const CameraManager& GetCameraManager() const; 
    CameraManager& GetCameraManager(); 

    Configuration& GetConfiguration();

    void StartCameraThread();
    void StopCameraThread();

    void ReadNewConfiguration(Configuration& config);


private:

    std::atomic<bool> _running_instance;

    Configuration config;

    CameraManager camera_manager;
    std::thread camera_thread;

    DataHandler data_handler;
    
    PayloadState state;
    RX_Queue rx_queue;
    TX_Queue tx_queue;

    std::mutex mtx;
    std::condition_variable cv_queue;

    // Thread Pool
    std::unique_ptr<ThreadPool> thread_pool;


    void SwitchToState(PayloadState new_state);

    void RunStartupHealthProcedures();
    void RetrieveInternalStates();

    void StopThreadPool();

    
};

#endif // PAYLOAD_HPP
