#ifndef ERRORS_HPP
#define ERRORS_HPP

#include <deque>


// Master list of all runtime error codes throughout the codebase 
// system-wide error logging 
// Circular buffer 


enum class ErrorCode // Error codes  
{
    // TODO must be extremely well documented 

    OK, // Can't be logged
    PLACEHOLDER,

    // Divide per subsystem

    // Task execution and queue 
    // Thread pool 
    // Camera subsystem
    // Telemetry
    // Communication

    // UART
    UART_OPEN_FAILED,
    UART_OPEN_FAILED_AFTER_RETRY,
    UART_GETATTR_FAILED,
    UART_CONFIG_FAILED,
    



    // Neural Engine 
    // OD


    UNDEFINED // Last error for checking (Sentinel value)
};

// Alias for readability
using EC = ErrorCode;


// Log any incoming error with their correspondign timestamp in the circular buffer 
// Note that OK is never logged to the buffer
void LogError(EC error_code);


// Retrieve the latest generated error. Default to OK if empty.
EC GetLastError();


// Get the number of errors in the buffer. If the buffer is full, this size will remain constant indefinitely.
std::size_t GetCurrentErrorCount();

// Simple getter to retrieve the nmx size of the buffer
std::size_t GetMaxErrorBufferSize();

// Clear the circular buffer
void ClearErrors();

#endif // ERRORS_HPP