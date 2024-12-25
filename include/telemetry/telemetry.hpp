#ifndef TELEMETRY_HPP
#define TELEMETRY_HPP

#include <mutex>
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

    long TIME;
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


    void RunService();

    // Returns a copy of the current telemetry frame 
    TelemetryFrame GetTmFrame() const;


    


private:

    TegraTM* shared_mem;
    TelemetryFrame tm_frame;
    mutable std::mutex frame_mtx;


    
    bool LinkToTegrastatsProcess();

    void UpdateFrame(Payload* payload);

};



int CountActiveThreads();

#endif // TELEMETRY_HPP