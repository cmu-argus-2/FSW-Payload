#include "vision/region_classifier.hpp"
#include <filesystem> 

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
