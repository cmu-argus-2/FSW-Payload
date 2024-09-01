#ifndef REGION_CLASSIFIER_HPP
#define REGION_CLASSIFIER_HPP

#include <torch/torch.h>
#include <string>


bool DetectGPU();

// Verify there is only one model file in the given directory
bool VerifySingleRcModel(const std::string& directory);




#endif // REGION_CLASSIFIER_HPP