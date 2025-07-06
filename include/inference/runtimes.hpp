#ifndef RUNTIMES_HPP
#define RUNTIMES_HPP

#include <string> 

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "spdlog/spdlog.h"

#include "inference/structures.hpp"
#include "core/errors.hpp"

namespace Inference
{


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

class RCNet
{

public:

    RCNet();
    ~RCNet();

    EC LoadEngine(const std::string& engine_path);
    void Infer(const void* input_data, void* output);


private:

    // Logger for TensorRT
    Logger trt_logger_;

    // Inference buffer 
    InferenceBuffer buffer_;

    // Model deserialization >> executable engine
    std::unique_ptr<IRuntime> runtime_ = nullptr;
    // Executable engine for inference (contains optimized computation graph)
    std::unique_ptr<ICudaEngine> engine_ = nullptr;
    // To execute the inference on a specific batch of inputs (binds to inputs/outputs buffers)
    std::unique_ptr<IExecutionContext> context_ = nullptr;

    // CUDA stream
    cudaStream_t stream_ = nullptr;

};


} // namespace Inference

#endif // RUNTIMES_HPP