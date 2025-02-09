/*

This file contains a series of utility functions to interface with the Real-Time Clock (RTC) and system time.
The time references are provided in milliseconds. 
 
Functions relies on the `clock_gettime` function from the POSIX standard to retrieve time values.
see: https://www.man7.org/linux/man-pages/man3/clock_gettime.3.html

 */
#ifndef TIMING_HPP
#define TIMING_HPP

#include <time.h>
#include <cstdint>

namespace timing
{
    
    // All in milliseconds
    
    // Time reference for the uptime of the system 
    extern uint64_t FSW_BOOT_TIME_REFERENCE;

    // Initialize the reference for the uptime of the system. Must be called first
    void InitializeBootTime();

    // Get time since the first call to InitializeBootTime()
    uint64_t GetUptime();

    uint64_t GetMonotonicTime();

    uint64_t GetCurrentTime();

    // void SetTimeRTC();

}





#endif // TIMING_HPP
