#include <string>
#include <cstdio>
#include "unistd.h"
#include "spdlog/spdlog.h"

#include "telemetry/telemetry.hpp"

bool CheckTegraTmProcessRunning()
{   
    std::string cmd = "pgrep " + std::string(TM_TEGRASTATS) + " > /dev/null 2>&1";
    // redirect stdout and stderr output to /dev/null 2>&1 (supress it)
    int result = std::system(cmd.c_str());
    // 0 --> A matching process was found.
    // 1 --> No matching process was found.
    return (result == 0); // Return true if the process is running
}

bool KillTegraTmProcess() 
{

    if (CheckTegraTmProcessRunning()) 
    {
        std::string cmd = std::string("pkill ") + TM_TEGRASTATS;
        int kill_res = std::system(cmd.c_str());
        return (kill_res == 0); // Return true if the kill command was successful
    }
    // Process was not running
    return false;
}


bool StartTegrastatsProcessor()
{
    // kill any process that was already running
    KillTegraTmProcess();
    
    std::string cmd = std::string("./build/bin/") + TM_TEGRASTATS;
    int result = std::system(cmd.c_str());

    return (result == 0);
}


TelemetryFrame::TelemetryFrame()
: 
TIME(0), 
UPTIME(0), 
PAYLOAD_STATE(0), 
ACTIVE_CAMERAS(0), 
CAPTURE_MODE(0),
CAM_STATUS{0,0,0,0},
TASKS_IN_EXECUTION(0), 
DISK_USAGE(0), 
LATEST_ERROR(0),
LAST_EXECUTED_CMD_ID(0), 
LAST_EXECUTED_CMD_TIME(0),
TEGRASTATS_PROCESS_STATUS(false), 
RAM_USAGE(0), 
SWAP_USAGE(0),
ACTIVE_CORES(0), 
CPU_LOAD{0, 0, 0, 0, 0, 0},
GPU_FREQ(0),
CPU_TEMP(0), 
GPU_TEMP(0), 
VDD_IN(0), 
VDD_CPU_GPU_CV(0), 
VDD_SOC(0)
{
}


Telemetry::Telemetry()
:
shared_mem(nullptr),
tm_frame()
{ 
}

TelemetryFrame Telemetry::GetTmFrame() const
{
    std::lock_guard<std::mutex> lock(frame_mtx);
    return tm_frame;
}




bool LinkToTegrastatsProcess()
{


    return true;


}

void UpdateFrame(Payload* payload)
{

    // System part 







    // Tegrastats part



    (void)payload;
}





int CountActiveThreads()
{
    char buffer[128];
    FILE* fp = popen(("ls -1 /proc/" + std::to_string(getpid()) + "/task | wc -l").c_str(), "r");

    if (!fp)
    {
        spdlog::error("Couldn't run active thread count command line");
        return -1; 
    }

    fgets(buffer, sizeof(buffer), fp);
    pclose(fp);

    return std::atoi(buffer);
}

