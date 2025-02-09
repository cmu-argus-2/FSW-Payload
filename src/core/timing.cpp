#include "core/timing.hpp"


namespace timing
{

    uint64_t FSW_BOOT_TIME_REFERENCE = 0;
    static bool is_initialized = false; // Guard flag
    
    void InitializeBootTime()
    {
        if (!is_initialized)  
        {
            FSW_BOOT_TIME_REFERENCE = GetMonotonicTime();
            is_initialized = true;
        }
    }

    uint64_t GetUptime()
    {
        if (!is_initialized)
        {
            InitializeBootTime();
        }
            
        return GetMonotonicTime() - FSW_BOOT_TIME_REFERENCE;
    }



    uint64_t GetMonotonicTime() 
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL;
    }


    uint64_t GetCurrentTime() 
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL;
    }


}


