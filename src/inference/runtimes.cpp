#include "inference/runtimes.hpp"

#include <vector>
#include <fstream>
#include <iostream>
namespace Inference
{

using namespace nvinfer1;

static constexpr const char* input_name = "input";
static constexpr const char* output_name = "output";


RCNet::RCNet()
{
}

RCNet::~RCNet()
{
    cudaStreamDestroy(stream_);

    buffers_.free();
    
    if (context_) 
    {
        context_.reset();
    }
    if (engine_) 
    {
        engine_.reset();
    }
    if (runtime_) 
    {
        runtime_.reset();
    }
}

EC RCNet::LoadEngine(const std::string& engine_path)
{
    // TODO: Absolutely need to add retry and recovery logic
    
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) 
    {
        spdlog::error("Error: Failed to open engine file {}", engine_path);
        return EC::NN_FAILED_TO_OPEN_ENGINE_FILE;
    }
    file.seekg(0, std::ios::end); // Move > end of file
    size_t size = file.tellg(); 
    file.seekg(0, std::ios::beg); // Move >  beginning of the file
    std::vector<char> engine_data(size); 
    file.read(engine_data.data(), size);

    // Create runtime and deserialize engine
    runtime_ = std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(trt_logger_));
    if (!runtime_) {
        spdlog::error("Error: Failed to create runtime");
        LogError(EC::NN_FAILED_TO_CREATE_RUNTIME);
        return EC::NN_FAILED_TO_CREATE_RUNTIME;
    }

    engine_ = std::unique_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) {
        spdlog::error("Error: Failed to create CUDA engine");
        LogError(EC::NN_FAILED_TO_CREATE_ENGINE);
        return EC::NN_FAILED_TO_CREATE_ENGINE;
    }

    // Create execution context
    context_ = std::unique_ptr<IExecutionContext>(engine_->createExecutionContext());
    if (!context_) 
    {
        spdlog::error("Error: Failed to create execution context");
        LogError(EC::NN_FAILED_TO_CREATE_EXECUTION_CONTEXT);
        return EC::NN_FAILED_TO_CREATE_EXECUTION_CONTEXT;
    }

    assert(engine_->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
    assert(engine_->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);

    context_->setInputShape(input_name, nvinfer1::Dims4{BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH});

    // Allocate GPU memory for input and output
    buffers_.input_size = BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
    buffers_.output_size = RC_NUM_CLASSES * sizeof(float);
    buffers_.allocate();

    // Bind input and output memory (ok for RC)
    context_->setTensorAddress(input_name, buffers_.input_data);
    context_->setTensorAddress(output_name, buffers_.output_data);

    // Creating CUDA stream for asynchronous execution
    cudaStreamCreate(&stream_); 

    initialized_ = true;

    return EC::OK;
}

EC RCNet::Infer(const void* input_data, void* output)
{
    
    if (!initialized_) 
    {
        spdlog::error("RCNet is not initialized. Call LoadEngine first.");
        return EC::NN_ENGINE_NOT_INITIALIZED;
    }

    if (!input_data || !output) 
    {
        spdlog::error("Input data or output buffer is null.");
        LogError(EC::NN_POINTER_NULL);
        return EC::NN_POINTER_NULL;
    }

    // copy input data to GPU
    // spdlog::info("Copying input data to GPU memory.");
    cudaError_t err = cudaMemcpy(buffers_.input_data, input_data, buffers_.input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        spdlog::error("cudaMemcpyAsync (input) failed.");
        return EC::NN_CUDA_MEMCPY_FAILED;
    }

    // Run asynchronous inference with enqueueV3 (async by design)
    bool status = context_->enqueueV3(stream_); // could use the default stream with enqueueV3(0) adn then no need to sync
    if (!status)
    {
        spdlog::error("Failed to enqueue inference.");
        LogError(EC::NN_INFERENCE_FAILED);
        return EC::NN_INFERENCE_FAILED;
    }

    err = cudaMemcpyAsync(output, buffers_.output_data, buffers_.output_size, cudaMemcpyDeviceToHost, stream_); // Copy output back to CPU
    if (err != cudaSuccess) 
    {
        spdlog::error("cudaMemcpyAsync output failed.");
        LogError(EC::NN_CUDA_MEMCPY_FAILED);
        return EC::NN_CUDA_MEMCPY_FAILED;
    }
    // spdlog::info("Inference enqueued successfully. Waiting for completion...");
    cudaStreamSynchronize(stream_);  
    return EC::OK;
}

// LD Net methods

// TODO: LDnet is commented out for compilation

// LDNet::LDNet(RegionID region_id)
// : region_id_(region_id)
// {
// }

// LDNet::~LDNet()
// {
//     cudaStreamDestroy(stream_);

//     buffers_.free();
    
//     if (context_) 
//     {
//         context_.reset();
//     }
//     if (engine_) 
//     {
//         engine_.reset();
//     }
//     if (runtime_) 
//     {
//         runtime_.reset();
//     }
// }

// EC LDNet::LoadEngine(const std::string& engine_path)
// {
//     // TODO: Absolutely need to add retry and recovery logic
    
//     std::ifstream file(engine_path, std::ios::binary);
//     if (!file.good()) 
//     {
//         spdlog::error("Error: Failed to open engine file {}", engine_path);
//         return EC::NN_FAILED_TO_OPEN_ENGINE_FILE;
//     }
//     file.seekg(0, std::ios::end); // Move > end of file
//     size_t size = file.tellg(); 
//     file.seekg(0, std::ios::beg); // Move >  beginning of the file
//     std::vector<char> engine_data(size); 
//     file.read(engine_data.data(), size);

//     // Create runtime and deserialize engine
//     runtime_ = std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(trt_logger_));
//     if (!runtime_) {
//         spdlog::error("Error: Failed to create runtime");
//         LogError(EC::NN_FAILED_TO_CREATE_RUNTIME);
//         return EC::NN_FAILED_TO_CREATE_RUNTIME;
//     }

//     engine_ = std::unique_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
//     if (!engine_) {
//         spdlog::error("Error: Failed to create CUDA engine");
//         LogError(EC::NN_FAILED_TO_CREATE_ENGINE);
//         return EC::NN_FAILED_TO_CREATE_ENGINE;
//     }

//     // Create execution context
//     context_ = std::unique_ptr<IExecutionContext>(engine_->createExecutionContext());
//     if (!context_) 
//     {
//         spdlog::error("Error: Failed to create execution context");
//         LogError(EC::NN_FAILED_TO_CREATE_EXECUTION_CONTEXT);
//         return EC::NN_FAILED_TO_CREATE_EXECUTION_CONTEXT;
//     }

//     assert(engine_->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
//     assert(engine_->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);

//     context_->setInputShape(input_name, nvinfer1::Dims4{BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH});

//     // Allocate GPU memory for input and output
//     buffers_.input_size = BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
//     buffers_.output_size = RC_NUM_CLASSES * sizeof(float);
//     buffers_.allocate();

//     // Bind input and output memory (ok for RC)
//     context_->setTensorAddress(input_name, buffers_.input_data);
//     context_->setTensorAddress(output_name, buffers_.output_data);

//     // Creating CUDA stream for asynchronous execution
//     cudaStreamCreate(&stream_); 

//     initialized_ = true;

//     return EC::OK;
// }

// EC LDNet::Infer(const void* input_data, void* output)
// {
    
//     if (!initialized_) 
//     {
//         spdlog::error("LDNet is not initialized. Call LoadEngine first.");
//         return EC::NN_ENGINE_NOT_INITIALIZED;
//     }

//     if (!input_data || !output) 
//     {
//         spdlog::error("Input data or output buffer is null.");
//         LogError(EC::NN_POINTER_NULL);
//         return EC::NN_POINTER_NULL;
//     }

//     // copy input data to GPU
//     // spdlog::info("Copying input data to GPU memory.");
//     cudaError_t err = cudaMemcpy(buffers_.input_data, input_data, buffers_.input_size, cudaMemcpyHostToDevice);
//     if (err != cudaSuccess) 
//     {
//         spdlog::error("cudaMemcpyAsync (input) failed.");
//         return EC::NN_CUDA_MEMCPY_FAILED;
//     }

//     // Run asynchronous inference with enqueueV3 (async by design)
//     bool status = context_->enqueueV3(stream_); // could use the default stream with enqueueV3(0) adn then no need to sync
//     if (!status)
//     {
//         spdlog::error("Failed to enqueue inference.");
//         LogError(EC::NN_INFERENCE_FAILED);
//         return EC::NN_INFERENCE_FAILED;
//     }

//     err = cudaMemcpyAsync(output, buffers_.output_data, buffers_.output_size, cudaMemcpyDeviceToHost, stream_); // Copy output back to CPU
//     if (err != cudaSuccess) 
//     {
//         spdlog::error("cudaMemcpyAsync output failed.");
//         LogError(EC::NN_CUDA_MEMCPY_FAILED);
//         return EC::NN_CUDA_MEMCPY_FAILED;
//     }
//     // spdlog::info("Inference enqueued successfully. Waiting for completion...");
//     cudaStreamSynchronize(stream_);  
//     return EC::OK;
// }


} // namespace Inference