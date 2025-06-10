#ifndef RUNTIMES_HPP
#define RUNTIMES_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "spdlog/spdlog.h"

#include "inference/structures.hpp"

using namespace nvinfer1;

// Logger for TensorRT
class Logger : public ILogger 
{
    void log(Severity severity, const char* msg) noexcept override 
    {
        if (severity <= Severity::kINFO)
        {
            spdlog::info("[TRT] {}", msg);
        }
        else if (severity == Severity::kWARNING)
        {
            spdlog::warn("[TRT] {}", msg);
        }
        else if (severity == Severity::kERROR)
        {
            spdlog::error("[TRT] {}", msg);
        }
        else if (severity == Severity::kINTERNAL_ERROR)
        {
            spdlog::critical("[TRT] {}", msg);
        }
    }
};



#endif // RUNTIMES_HPP