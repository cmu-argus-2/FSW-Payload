#include "vision/region_classifier.hpp"
#include <filesystem> 
#include "spdlog/spdlog.h"


bool DetectGPU() {
    return torch::cuda::is_available();
}

bool VerifySingleRcModel(const std::string& directory) {
    namespace fs = std::filesystem;

    size_t count = 0;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            count++;
        }
    }
    // Return true if there is exactly one file in the directory
    return count == 1;
}


RegionClassifierModel::RegionClassifierModel(const std::string& model_path) 
: device(torch::kCPU) 
{
    load_model(model_path);
    set_device();
}

void RegionClassifierModel::load_model(const std::string& model_path) 
{
    try {
        model = torch::jit::load(model_path);
        SPDLOG_INFO("Model loaded successfully from {}", model_path);
    } catch (const c10::Error& e) {
        SPDLOG_ERROR("Error loading model: {}", e.what_without_backtrace());
        // TODO: Add error handling
        throw;
    }
}

void RegionClassifierModel::set_device() {
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        SPDLOG_INFO("Using GPU CUDA");
    } else {
        device = torch::Device(torch::kCPU);
        SPDLOG_INFO("Using CPU");
    }
    model.to(device);
}

torch::Tensor RegionClassifierModel::run_inference(torch::Tensor input) {
    input = input.to(device);  
    return model.forward({input}).toTensor();
}