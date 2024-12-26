#include <string>
#include <cstdio>
#include <chrono>
#include <thread>
#include <sys/mman.h>
#include <unistd.h>

#include "spdlog/spdlog.h"

#include "payload.hpp"
#include "core/data_handling.hpp"
#include "telemetry/telemetry.hpp"
#include "core/utils.hpp"

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


bool RestartTegrastatsProcessor()
{
    // kill any process that was already running
    KillTegraTmProcess();
    
    std::string cmd = std::string("./build/bin/") + TM_TEGRASTATS;

    if (!DetectJetsonPlatform()) // for emulation
    {   
        cmd += " emulate";
    }

    // Add redirection and background execution
    cmd += " > /dev/null 2>&1 &";


    int result = std::system(cmd.c_str());

    return (result == 0);
}


TelemetryFrame::TelemetryFrame()
: 
SYSTEM_TIME(0), 
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

void PrintTelemetryFrame(const TelemetryFrame& tm_frame)
{
    SPDLOG_INFO("SYSTEM_TIME: {}", tm_frame.SYSTEM_TIME);
    SPDLOG_INFO("UPTIME: {}", tm_frame.UPTIME);
    SPDLOG_INFO("PAYLOAD_STATE: {}", tm_frame.PAYLOAD_STATE);
    SPDLOG_INFO("ACTIVE_CAMERAS: {}", tm_frame.ACTIVE_CAMERAS);
    SPDLOG_INFO("CAPTURE_MODE: {}", tm_frame.CAPTURE_MODE);
    SPDLOG_INFO("CAM_STATUS: [{}, {}, {}, {}]", tm_frame.CAM_STATUS[0], tm_frame.CAM_STATUS[1], tm_frame.CAM_STATUS[2], tm_frame.CAM_STATUS[3]);
    SPDLOG_INFO("TASKS_IN_EXECUTION: {}", tm_frame.TASKS_IN_EXECUTION);
    SPDLOG_INFO("DISK_USAGE: {}", tm_frame.DISK_USAGE);
    SPDLOG_INFO("LATEST_ERROR: {}", tm_frame.LATEST_ERROR);
    SPDLOG_INFO("LAST_EXECUTED_CMD_ID: {}", tm_frame.LAST_EXECUTED_CMD_ID);
    SPDLOG_INFO("LAST_EXECUTED_CMD_TIME: {}", tm_frame.LAST_EXECUTED_CMD_TIME);
    SPDLOG_INFO("TEGRASTATS_PROCESS_STATUS: {}", tm_frame.TEGRASTATS_PROCESS_STATUS);
    SPDLOG_INFO("RAM_USAGE: {}", tm_frame.RAM_USAGE);
    SPDLOG_INFO("SWAP_USAGE: {}", tm_frame.SWAP_USAGE);
    SPDLOG_INFO("ACTIVE_CORES: {}", tm_frame.ACTIVE_CORES);
    SPDLOG_INFO("CPU_LOAD: [{}, {}, {}, {}, {}, {}]", tm_frame.CPU_LOAD[0], tm_frame.CPU_LOAD[1], tm_frame.CPU_LOAD[2], tm_frame.CPU_LOAD[3], tm_frame.CPU_LOAD[4], tm_frame.CPU_LOAD[5]);
    SPDLOG_INFO("GPU_FREQ: {}", tm_frame.GPU_FREQ);
    SPDLOG_INFO("CPU_TEMP: {}", tm_frame.CPU_TEMP);
    SPDLOG_INFO("GPU_TEMP: {}", tm_frame.GPU_TEMP);
    SPDLOG_INFO("VDD_IN: {}", tm_frame.VDD_IN);
    SPDLOG_INFO("VDD_CPU_GPU_CV: {}", tm_frame.VDD_CPU_GPU_CV);
    SPDLOG_INFO("VDD_SOC: {}", tm_frame.VDD_SOC);
}

Telemetry::Telemetry()
:
shared_mem(nullptr),
sem_shared_mem(nullptr),
tm_frame()
{ 
    if (!CheckTegraTmProcessRunning())
    {
        if (RestartTegrastatsProcessor())
        {
            SPDLOG_INFO("Started TM Tegrastats Processor");
        }
        else
        {
            SPDLOG_WARN("Failed to start the TM Tegrastats Processor");
        }
    }

    if (LinkToTegraTmProcess())
    {
        SPDLOG_INFO("Linked shared memory to tegrastats processor");
    }
    else
    {
        SPDLOG_WARN("Failed to link shared memory to tegrastats processor");
    }

}


Telemetry::~Telemetry()
{
    StopService();
    
    KillTegraTmProcess();

    if (sem_shared_mem != nullptr)
    {
        sem_close(sem_shared_mem);
        sem_shared_mem = nullptr;
    }

    if (shared_mem != nullptr)
    {
        munmap(shared_mem, sizeof(shared_mem));
        shared_mem = nullptr;
    }
}

TelemetryFrame Telemetry::GetTmFrame() const
{
    std::lock_guard<std::mutex> lock(frame_mtx);
    return tm_frame;
}


bool Telemetry::LinkToTegraTmProcess() 
{   
    // if already configured, return immediately
    if (tegra_tm_configured)
    {
        return tegra_tm_configured;
    }


    if (!LinkToSharedMemory(shared_mem)) {
        SPDLOG_ERROR("Failed to link to shared memory");
        tegra_tm_configured = false;
        return false;
    }

    // close existing semaphore if already open
    if (sem_shared_mem != nullptr) {
        sem_close(sem_shared_mem);
        sem_shared_mem = nullptr;
    }


    sem_shared_mem = LinkToSemaphore();
    if (sem_shared_mem == nullptr) 
    {
        SPDLOG_ERROR("Failed to link to semaphore");
        tegra_tm_configured = false;
        return false;
    }

    return true;
}


void Telemetry::UpdateFrame(Payload* payload)
{

    // System part 
    _UpdateTmSystemPart(payload);

    // Tegrastats part
    bool update = _UpdateTmTegraPart();
    if (!update)
    {
        _counter_before_tegra_restart++;
        if (_counter_before_tegra_restart >= MAXIMUM_COUNT_WITHOUT_UPDATE)
        {
            SPDLOG_WARN("Restarting TM_TEGRASTATS...");
            RestartTegrastatsProcessor();
            LinkToTegraTmProcess();
            _counter_before_tegra_restart = 0;
        }
    }

}

void Telemetry::RunService(Payload* payload)
{

    loop_flag.store(true); // just to be explicit
    const auto service_start = std::chrono::high_resolution_clock::now();
    const auto interval = std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
        std::chrono::duration<double>(1.0 / tm_frequency_hz)
    );
    auto next = std::chrono::high_resolution_clock::now();

    while (loop_flag.load()) 
    {

        UpdateFrame(payload);

        PrintTelemetryFrame(tm_frame);


        next += interval;
        std::this_thread::sleep_until(next);
    }


    (void)payload;

}

void Telemetry::StopService()
{
    loop_flag.store(false);
    // join externally
}


void Telemetry::_UpdateTmSystemPart(Payload* payload)
{
    SPDLOG_DEBUG("Updating system part of the TM frame..");
    // TODO error handling

    // tm_frame.SYSTEM_TIME = 
    // tm_frame.SYSTEM_UPTIME = 


    tm_frame.PAYLOAD_STATE = static_cast<uint8_t>(payload->GetState());
    tm_frame.ACTIVE_CAMERAS = static_cast<uint8_t>(payload->GetCameraManager().CountActiveCameras());
    tm_frame.CAPTURE_MODE = static_cast<uint8_t>(payload->GetCameraManager().GetCaptureMode());
    payload->GetCameraManager().FillCameraStatus(tm_frame.CAM_STATUS);
    int disk_use = DH::GetTotalDiskUsage();
    if (disk_use >= 0)
    {
        tm_frame.DISK_USAGE = static_cast<uint8_t>(disk_use);
    }
    else
    {
        tm_frame.DISK_USAGE = 0;
    }

    // tm_frame.LATEST_ERROR = 
    // tm_frame.LAST_EXECUTED_CMD_ID = 
    
}


bool Telemetry::_UpdateTmTegraPart() 
{
    SPDLOG_DEBUG("Updating tegra part of the TM frame..");
    
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += SEMAPHORE_TIMEOUT_NS; // Add the timeout 
    // need to deal with overflow otherwise will cause undefined behavior at the sem_timedwait
    if (ts.tv_nsec >= 1000000000)
    {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000;
    }

    if (sem_timedwait(sem_shared_mem, &ts) == -1) 
    {
        if (errno == ETIMEDOUT) 
        {
            SPDLOG_ERROR("Semaphore timeout occurred!");
            // Skip 
            return false;

        } else {
            SPDLOG_ERROR("Semaphore error occurred: {}", strerror(errno));
            // Exit 
            return false;
        }
    }

    // Semaphore On

    if (shared_mem->change_flag == 0)
    {
        sem_post(sem_shared_mem); // Unlock access
        SPDLOG_INFO("Tegrastats process hasn't updated the shared memory yet.");
        return false;
    }

   
    // Copy shared memory contents into TM frame

    tm_frame.RAM_USAGE = static_cast<uint8_t>((shared_mem->ram_used * 100.0f) / shared_mem->ram_total);
    tm_frame.SWAP_USAGE = static_cast<uint8_t>((shared_mem->swap_used * 100.0f) / shared_mem->swap_total);
    tm_frame.ACTIVE_CORES = static_cast<uint8_t>(shared_mem->active_cores);
    std::copy(std::begin(shared_mem->cpu_load), std::end(shared_mem->cpu_load), std::begin(tm_frame.CPU_LOAD));
    tm_frame.GPU_FREQ = shared_mem->gpu_freq;
    tm_frame.CPU_TEMP = static_cast<uint8_t>(shared_mem->cpu_temp); 
    tm_frame.GPU_TEMP = static_cast<uint8_t>(shared_mem->gpu_temp); 
    tm_frame.VDD_IN = shared_mem->vdd_in;
    tm_frame.VDD_CPU_GPU_CV = shared_mem->vdd_cpu_gpu_cv;
    tm_frame.VDD_SOC = shared_mem->vdd_soc;

    // Set the read flag to 0
    memcpy(&(shared_mem->change_flag), &read_flag, sizeof(shared_mem->change_flag));
    sem_post(sem_shared_mem); // Unlock access

    return true;
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

