/*

This file contains a series of utility functions to interface with the Real-Time Clock (RTC) and system time.
The time references are provided in milliseconds. 
 
Functions relies on the `clock_gettime` function from the POSIX standard to retrieve time values.
see: https://www.man7.org/linux/man-pages/man3/clock_gettime.3.html


Unix Epoch: the time 00:00:00 UTC on 1 January 1970 (or 1970-01-01T00:00:00Z ISO 8601). 
NOTE: We don't need dates since we're using a UNIX timestamp

 */
#ifndef TIMING_HPP
#define TIMING_HPP

#include <cstdint>
#include <chrono>

namespace timing
{
    using Clock = std::chrono::high_resolution_clock; 
    
    // All in milliseconds
    
    // Time reference for the uptime of the system 
    extern uint64_t FSW_BOOT_TIME_REFERENCE;

    // Initialize the reference for the uptime of the system. Must be called first
    void InitializeBootTime();

    // Get time since the first call to InitializeBootTime()
    uint64_t GetUptimeMs();

    uint64_t GetMonotonicTimeMs();

    uint64_t GetCurrentTimeMs();

    void SleepMs(uint32_t milliseconds);

    void SleepSeconds(uint32_t seconds);

    // void SetTimeRTC();

}





#endif // TIMING_HPP
