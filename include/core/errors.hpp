#ifndef ERRORS_HPP
#define ERRORS_HPP

#include <deque>
#include <cstdint>


// Master list of all runtime error codes throughout the codebase 
// system-wide error logging 
// Circular buffer 


// TODO: constexpr
enum class ErrorCode // Error codes  
{
    // TODO must be extremely well documented 
    OK = 0, // Can't be logged
    PLACEHOLDER = 1,
    // Commands
    INVALID_COMMAND_ID,
    INVALID_COMMAND_ARGUMENTS,
    NO_FILE_READY,
    NO_MORE_PACKET_FOR_FILE, 
    FAIL_TO_READ_FILE, 
    FILE_NOT_AVAILABLE, // DO NOT CHANGE ALL ABOVE

    // Data handling
    FILE_DOES_NOT_EXIST,
    FILE_NOT_FOUND,
    START_BYTE_OUT_OF_RANGE,
    FAILED_TO_GRAB_FILE_CHUNK,
    
    // Camera subsystem
    CAMERA_CAPTURE_FAILED,
    CAMERA_INITIALIZATION_FAILED,


    // Task execution and queue 
    // Thread pool 



    // Telemetry


    // UART
    UART_OPEN_FAILED,
    UART_OPEN_FAILED_AFTER_RETRY,
    UART_NOT_OPEN,
    UART_GETATTR_FAILED,
    UART_SETATTR_FAILED,
    UART_FAILED_WRITE,
    UART_WRITE_BUFFER_OVERFLOW,
    UART_INCOMPLETE_READ,
    



    // Neural Engine 
    NN_FAILED_TO_OPEN_ENGINE_FILE,
    NN_FAILED_TO_CREATE_RUNTIME,
    NN_FAILED_TO_CREATE_ENGINE,
    NN_FAILED_TO_CREATE_EXECUTION_CONTEXT,
    NN_FAILED_TO_LOAD_ENGINE,
    NN_ENGINE_NOT_INITIALIZED, // Runtime not initialized
    NN_CUDA_MEMCPY_FAILED, 
    NN_POINTER_NULL, 
    NN_INFERENCE_FAILED, 
    NN_NO_FRAME_AVAILABLE, // No frame available for inference



    // OD


    UNDEFINED // Last error for checking (Sentinel value)
};

// Alias for readability
using EC = ErrorCode;


constexpr uint8_t to_uint8(ErrorCode ec) 
{
    return static_cast<uint8_t>(ec);
}


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