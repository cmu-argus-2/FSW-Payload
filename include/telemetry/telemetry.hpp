#ifndef MONITORING_HPP
#define MONITORING_HPP


#define SEMAPHORE_TIMEOUT_NS 500000000 // 500 milliseconds


struct TelemetryFrame
{

    long TIME;
    long UPTIME;
    uint8_t PAYLOAD_STATE;
    uint8_t ACTIVE_CAMERAS;
    uint8_t CAPTURE_MODE;
    uint8_t CAM1_STATUS;
    uint8_t CAM2_STATUS;
    uint8_t CAM3_STATUS;
    uint8_t CAM4_STATUS;
    uint8_t TASKS_IN_EXECUTION;
    uint8_t DISK_USAGE;
    uint16_t SESSION_ERROR_COUNT;
    uint8_t LATEST_ERROR;
    uint8_t LAST_EXECUTED_CMD_ID;
    long LAST_EXECUTED_CMD_TIME;
    bool TEGRASTATS_PROCESS_STATUS;
    uint8_t RAM_USAGE;
    uint8_t SWAP_USAGE;
    uint8_t ACTIVE_CORES;
    uint8_t CPU_LOAD_1;
    uint8_t CPU_LOAD_2;
    uint8_t CPU_LOAD_3;
    uint8_t CPU_LOAD_4;
    uint8_t CPU_LOAD_5;
    uint8_t CPU_LOAD_6;
    uint8_t GPU_FREQ;
    uint8_t CPU_TEMP;
    uint8_t GPU_TEMP;
    int VDD_IN;
    int VDD_CPU_GPU_CV;
    int VDD_SOC;
        

};



// int CountActiveThreads();


#endif // MONITORING_HPP