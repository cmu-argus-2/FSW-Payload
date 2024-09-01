
#ifndef REGION_CLASSIFIER_HPP
#define REGION_CLASSIFIER_HPP

#include <torch/script.h> 
#include <torch/torch.h>
#include <string>

#define NUM_CLASSES 16


bool DetectGPU();

// Verify there is only one model file in the given directory
bool VerifySingleRcModel(const std::string& directory);


class RegionClassifierModel 
{
public:
    RegionClassifierModel(const std::string& model_path);
    torch::Tensor run_inference(torch::Tensor input);

private:
    torch::jit::script::Module model;
    torch::Device device;

    void load_model(const std::string& model_path);
    void set_device();
};

#endif // REGION_CLASSIFIER_HPP