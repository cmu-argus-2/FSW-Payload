#include "vision/rc.hpp"
#include <filesystem> 
#include "spdlog/spdlog.h"

namespace RegionMapping 
{
    // Static mappings
    const std::unordered_map<std::string, RegionID> region_to_id = 
    {
        {"10S", 1}, {"10T", 2}, {"11R", 3}, {"12R", 4},
        {"16T", 5}, {"17R", 6}, {"17T", 7}, {"18S", 8},
        {"32S", 9}, {"32T", 10}, {"33S", 11}, {"33T", 12},
        {"52S", 13}, {"53S", 14}, {"54S", 15}, {"54T", 16}
    };

    const std::unordered_map<RegionID, std::string> id_to_region = 
    {
        {1, "10S"}, {2, "10T"}, {3, "11R"}, {4, "12R"},
        {5, "16T"}, {6, "17R"}, {7, "17T"}, {8, "18S"},
        {9, "32S"}, {10, "32T"}, {11, "33S"}, {12, "33T"},
        {13, "52S"}, {14, "53S"}, {15, "54S"}, {16, "54T"}
    };

    const std::unordered_map<RegionID, std::string> id_to_location = 
    {
        {1, "California"}, {2, "Washington / Oregon"}, {3, "Baja California, Mexico"},
        {4, "Sonora, Mexico"}, {5, "Minnesota / Wisconsin / Iowa / Illinois"},
        {6, "Florida"}, {7, "Toronto, Canada / Michigan / OH / PA"},
        {8, "New Jersey / Washington DC"}, {9, "Tunisia (North Africa near Tyrrhenian Sea)"},
        {10, "Switzerland / Italy / Tyrrhenian Sea"}, {11, "Sicilia, Italy"},
        {12, "Italy / Adriatic Sea"}, {13, "Korea / Kumamoto, Japan"},
        {14, "Hiroshima to Nagoya, Japan"}, {15, "Tokyo to Hachinohe, Japan"},
        {16, "Sapporo, Japan"}
    };

    RegionID GetRegionID(const std::string& region) 
    {
        auto it = region_to_id.find(region);
        return (it != region_to_id.end()) ? it->second : 0; // 0 = Invalid
    }

    std::string GetRegionString(RegionID id)
    {
        auto it = id_to_region.find(id);
        return (it != id_to_region.end()) ? it->second : "UNKNOWN";
    }

    std::string GetRegionLocation(RegionID id) 
    {
        auto it = id_to_location.find(id);
        return (it != id_to_location.end()) ? it->second : "UNKNOWN";
    }
}

#if NN_ENABLED
bool DetectGPU() {
    return torch::cuda::is_available();
}

bool VerifySingleRcModel(const std::string& directory) {
    namespace fs = std::filesystem;

    size_t count = 0;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            ++count;
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

#endif // NN_ENABLED