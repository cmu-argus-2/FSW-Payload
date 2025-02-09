#ifndef TELEMETRY_HPP
#define TELEMETRY_HPP

#include <atomic>
#include <mutex>
#include <semaphore.h>
#include "telemetry/tegra.hpp"

#define SEMAPHORE_TIMEOUT_NS 200000000 // 200 milliseconds
#define DEFAULT_TM_FREQUENCY_HZ 1  

#define MAXIMUM_COUNT_WITHOUT_UPDATE 3


// Execute the pgrep command and check its return value
bool CheckTegraTmProcessRunning();
// Kill the process
bool KillTegraTmProcess();
// (Re)start the TM_TEGRASTSTS executable that updates the shared memory
// No point in just starting
bool RestartTegrastatsProcessor();

struct TelemetryFrame
{
    // No need to optimize for alignment 

    // System part
    uint64_t SYSTEM_TIME;
    uint32_t SYSTEM_UPTIME;
    uint32_t LAST_EXECUTED_CMD_TIME;
    uint8_t LAST_EXECUTED_CMD_ID;
    uint8_t PAYLOAD_STATE;
    uint8_t ACTIVE_CAMERAS;
    uint8_t CAPTURE_MODE;
    uint8_t CAM_STATUS[4];
    uint8_t TASKS_IN_EXECUTION;
    uint8_t DISK_USAGE;
    uint8_t LATEST_ERROR;

    // Tegrastats part
    bool TEGRASTATS_PROCESS_STATUS;
    uint8_t RAM_USAGE;
    uint8_t SWAP_USAGE;
    uint8_t ACTIVE_CORES;
    uint8_t CPU_LOAD[6];
    uint8_t GPU_FREQ;
    uint8_t CPU_TEMP;
    uint8_t GPU_TEMP;
    // Blocks’s current power consumption in milliwatts 
    uint16_t VDD_IN; 
    uint16_t VDD_CPU_GPU_CV;
    uint16_t VDD_SOC;
    
    TelemetryFrame();
};

void PrintTelemetryFrame(const TelemetryFrame& tm_frame);

class Telemetry
{

public:


    Telemetry();

    ~Telemetry();

    // Runs the continuous update of the frame and the live management of the associated tegrastats process
    void RunService();
    // Stop the telemetry service (running on a thread)
    void StopService();

    // Returns a copy of the current telemetry frame 
    TelemetryFrame GetTmFrame() const;


    


private:

    int tm_frequency_hz = DEFAULT_TM_FREQUENCY_HZ;
    std::atomic<bool> loop_flag = false;

    uint8_t read_flag = 0; // reader set the flag to 0
    TegraTM* shared_mem;
    sem_t* sem_shared_mem;
    TelemetryFrame tm_frame;
    mutable std::mutex frame_mtx;
    
    // Link the shared memory and semaphore to those of the TM_TEGRASTASTS 
    bool LinkToTegraTmProcess();
    bool tegra_tm_configured = false;

    void UpdateFrame();

    void _UpdateTmSystemPart();
    // Returns True if successfully updated, False otherwise (semaphore failure, no tegra update, etc)
    bool _UpdateTmTegraPart();


    int _counter_before_tegra_restart = 0;


};



int CountActiveThreads();

#endif // TELEMETRY_HPP