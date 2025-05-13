#include <core/errors.hpp>
#include <mutex>
#include "spdlog/spdlog.h"


// Circular buffer as a deque
static std::deque<EC> _error_code_buffer;
constexpr std::size_t MAX_ERROR_CODE_BUFFER_SIZE = 20;
static std::mutex _error_code_buffer_mtx;

void LogError(EC error_code)
{
    // Validate error code with a sentinel
    assert(static_cast<int>(error_code) >= static_cast<int>(EC::OK) && 
           static_cast<int>(error_code) < static_cast<int>(EC::UNDEFINED) && 
           "Error code doesn't exist.");

    // Assert it's not OK 
    assert(error_code != EC::OK && "Can't log OK as an error.");

    std::lock_guard<std::mutex> lock(_error_code_buffer_mtx);

    if (_error_code_buffer.size() >= MAX_ERROR_CODE_BUFFER_SIZE)
    {
        _error_code_buffer.pop_front();
    }
    _error_code_buffer.push_back(error_code);

    SPDLOG_DEBUG("Logged Error Code: {}", error_code);
}


EC GetLastError()
{
    std::lock_guard<std::mutex> lock(_error_code_buffer_mtx);
    if (_error_code_buffer.empty())
    {
        return EC::OK; // default to OK if empty
    }

    return _error_code_buffer.back();
}


std::size_t GetCurrentErrorCount() 
{
    std::lock_guard<std::mutex> lock(_error_code_buffer_mtx);
    return _error_code_buffer.size();
}

std::size_t GetMaxErrorBufferSize()
{
    return MAX_ERROR_CODE_BUFFER_SIZE;
}

void ClearErrors()
{
    std::lock_guard<std::mutex> lock(_error_code_buffer_mtx);
    _error_code_buffer.clear();
}
