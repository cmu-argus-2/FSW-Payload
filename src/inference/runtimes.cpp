#include "inference/runtimes.hpp"

#include <vector>
#include <fstream>
#include <iostream>
namespace Inference
{

using namespace nvinfer1;

static constexpr const char* input_name = "l_x_";
static constexpr const char* output_name = "sigmoid_1";

static constexpr int BATCH_SIZE = 1; // Batch size for inference
static constexpr int INPUT_CHANNELS = 3; // Number of input channels (RGB)
static constexpr int INPUT_HEIGHT = 224; // Input height
static constexpr int INPUT_WIDTH = 224; // Input width
static constexpr int NUM_CLASSES = 16; // Number of classes in the output



RCNet::RCNet()
{
}

RCNet::~RCNet()
{
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
        return EC::NN_FAILED_TO_CREATE_RUNTIME;
    }

    engine_ = std::unique_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) {
        spdlog::error("Error: Failed to create CUDA engine");
        return EC::NN_FAILED_TO_CREATE_ENGINE;
    }

    // Create execution context
    context_ = std::unique_ptr<IExecutionContext>(engine_->createExecutionContext());
    if (!context_) 
    {
        spdlog::error("Error: Failed to create execution context");
        return EC::NN_FAILED_TO_CREATE_EXECUTION_CONTEXT;
    }

    assert(engine_->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
    assert(engine_->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);

    context_->setInputShape(input_name, nvinfer1::Dims4{BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH});

    // Allocate GPU memory for input and output
    buffer_.input_size = BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
    buffer_.output_size = NUM_CLASSES * sizeof(float);
    buffer_.allocate();

    // Bind input and output memory (ok for RC)
    context_->setTensorAddress(input_name, buffer_.input_data);
    context_->setTensorAddress(output_name, buffer_.output_data);

    // Creating CUDA stream for asynchronous execution
    cudaStreamCreate(&stream_); 

    return EC::OK;
}

void RCNet::Infer(const void* input_data, void* output)
{
    
    if (!context_ || !stream_ || !input_data || !output) 
    {
        spdlog::error("Inference called with null context, stream, or buffers.");
        return;
    }

    // copy input data to GPU
    cudaError_t err = cudaMemcpy(buffer_.input_data, input_data, buffer_.input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        spdlog::error("cudaMemcpyAsync (input) failed.");
        return;
    }

    // Run asynchronous inference 
    bool status = context_->enqueueV3(stream_);
    if (!status)
    {
        spdlog::error("Failed to enqueue inference.");
        return;
    }

    err = cudaMemcpyAsync(output, buffer_.output_data, buffer_.output_size, cudaMemcpyDeviceToHost, stream_); // Copy output back to CPU
    if (err != cudaSuccess) 
    {
        spdlog::error("cudaMemcpyAsync output failed.");
        return;
    }

    cudaStreamSynchronize(stream_);  
}



} // namespace Inference