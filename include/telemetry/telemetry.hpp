#ifndef TELEMETRY_HPP
#define TELEMETRY_HPP

#define SEMAPHORE_TIMEOUT_NS 100000000 // 100 milliseconds
#define DEFAULT_TM_FREQUENCY_HZ 1  

#include <mutex>
#include <semaphore.h>
#include "telemetry/tegra.hpp"


// Forward declaration
class Payload;


// Execute the pgrep command and check its return value
bool CheckTegraTmProcessRunning();
// Kill the process
bool KillTegraTmProcess();
// Start the TM_TEGRASTSTS executable that updates the shared memory
bool StartTegrastatsProcessor();

struct TelemetryFrame
{

    long SYSTEM_TIME;
    long UPTIME;
    uint8_t PAYLOAD_STATE;
    uint8_t ACTIVE_CAMERAS;
    uint8_t CAPTURE_MODE;
    uint8_t CAM_STATUS[4];
    uint8_t TASKS_IN_EXECUTION;
    uint8_t DISK_USAGE;
    uint8_t LATEST_ERROR;
    uint8_t LAST_EXECUTED_CMD_ID;
    long LAST_EXECUTED_CMD_TIME;
    bool TEGRASTATS_PROCESS_STATUS;
    uint8_t RAM_USAGE;
    uint8_t SWAP_USAGE;
    uint8_t ACTIVE_CORES;
    uint8_t CPU_LOAD[6];
    uint8_t GPU_FREQ;
    uint8_t CPU_TEMP;
    uint8_t GPU_TEMP;
    int VDD_IN;
    int VDD_CPU_GPU_CV;
    int VDD_SOC;
    
    TelemetryFrame();

};


class Telemetry
{

public:


    Telemetry();

    // Runs the continuous update of the frame and the live management of the associated tegrastats process
    void RunService(Payload* payload);

    // Returns a copy of the current telemetry frame 
    TelemetryFrame GetTmFrame() const;


    


private:

    int frequency_hz = DEFAULT_TM_FREQUENCY_HZ;

    uint8_t read_flag = 0; // reader set the flag to 0
    TegraTM* shared_mem;
    sem_t* sem_shared_mem;
    TelemetryFrame tm_frame;
    mutable std::mutex frame_mtx;
    
    // Link the shared memory and semaphore to those of the TM_TEGRASTASTS 
    bool LinkToTegraTmProcess();

    void UpdateFrame(Payload* payload);

    void _UpdateTmSystemPart(Payload* payload);
    // Returns True if successfully updated, False otherwise (semaphore failure, no tegra update, etc)
    bool _UpdateTmTegraPart();

};



int CountActiveThreads();

#endif // TELEMETRY_HPP