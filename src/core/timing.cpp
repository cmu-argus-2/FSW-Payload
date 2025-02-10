#include "core/timing.hpp"
#include <unistd.h>
#include <chrono>

namespace timing
{

    uint64_t FSW_BOOT_TIME_REFERENCE = 0;
    static bool is_initialized = false; // Guard flag
    
    void InitializeBootTime()
    {
        if (!is_initialized)  
        {
            FSW_BOOT_TIME_REFERENCE = GetMonotonicTimeMs();
            is_initialized = true;
        }
    }

    uint64_t GetUptimeMs()
    {
        if (!is_initialized)
        {
            InitializeBootTime();
        }
            
        return GetMonotonicTimeMs() - FSW_BOOT_TIME_REFERENCE;
    }



    uint64_t GetMonotonicTimeMs() 
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL;

        // Chrono version
        // auto now = std::chrono::steady_clock::now();
        // return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    }


    uint64_t GetCurrentTimeMs() 
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL;

        // Chrono version
        // auto now = std::chrono::system_clock::now();
        // return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    }

    void SleepMs(uint32_t milliseconds)
    {
        usleep(milliseconds * 1000);
    }

    void SleepSeconds(uint32_t seconds)
    {
        usleep(seconds * 1000000);
    }


}


