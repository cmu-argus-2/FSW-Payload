#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include <cuda_runtime_api.h>

namespace Inference
{

static constexpr int FULL_RES_WIDTH = 4608;
static constexpr int FULL_RES_HEIGHT = 2592;
static constexpr int DOWNSCALED_RES_WIDTH = 224;
static constexpr int DOWNSCALED_RES_HEIGHT = 224;

class InferenceEngine
{
public:
    virtual void infer() = 0; 
    virtual ~InferenceEngine() = 0;
};

// Inference buffer struct to be reused for all runtimes
struct InferenceBuffer
{
    void* input_data = nullptr;
    void* output_data = nullptr;

    size_t input_size = 0;
    size_t output_size = 0;

    void allocate()
    {
        // Will need to be carefull for x86 vs Tegra
        cudaMalloc(&input_data, input_size);
        cudaMalloc(&output_data, output_size);
    }

    void free()
    {
        if (input_data) 
        {
            cudaFree(input_data);
            input_data = nullptr;
        }
        if (output_data) 
        {
            cudaFree(output_data);
            output_data = nullptr;
        }
    }
};


} // namespace Inference

#endif // STRUCTURES_HPP