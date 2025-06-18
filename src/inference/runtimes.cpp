#include "inference/runtimes.hpp"

#include <vector>
#include <fstream>
#include <iostream>
namespace Inference
{

using namespace nvinfer1;

RCNet::RCNet()
{}

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
    runtime_ = std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(logger));
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

    return EC::OK;
}


} // namespace Inference