#ifndef RUNTIMES_HPP
#define RUNTIMES_HPP

#include <string> 

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <eigen3/Eigen/Dense>

#include "spdlog/spdlog.h"

#include "inference/structures.hpp"
#include "core/errors.hpp"
#include "vision/regions.hpp"
#include "vision/frame.hpp"

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

static constexpr int BATCH_SIZE = 1; // Batch size for inference
static constexpr int INPUT_CHANNELS = 3; // Number of input channels (RGB)
static constexpr int INPUT_HEIGHT = 224; // Input height
static constexpr int INPUT_WIDTH = 224; // Input width
static constexpr int RC_NUM_CLASSES = 16; // Number of classes in the output

class RCNet
{

public:

    RCNet();
    ~RCNet();

    EC LoadEngine(const std::string& engine_path);
    bool IsInitialized() const { return initialized_; }
    EC Infer(const void* input_data, void* output);


private:

    bool initialized_ = false; 

    // Logger for TensorRT
    Logger trt_logger_;

    // Inference buffer 
    InferenceBuffer buffers_;

    // Model deserialization >> executable engine
    std::unique_ptr<IRuntime> runtime_ = nullptr;
    // Executable engine for inference (contains optimized computation graph)
    std::unique_ptr<ICudaEngine> engine_ = nullptr;
    // To execute the inference on a specific batch of inputs (binds to inputs/outputs buffers)
    std::unique_ptr<IExecutionContext> context_ = nullptr;

    // CUDA stream
    cudaStream_t stream_ = nullptr;

};


class LDNet
{

public:

    LDNet(RegionID region_id, std::string csv_path);
    ~LDNet();

    EC LoadEngine(const std::string& engine_path);
    bool IsInitialized() const { return initialized_; }
    EC Infer(const void* input_data, void* output);
    int GetNumLandmarks() const { return num_landmarks_; }
    int GetOutputSize() const { return output_size_; }
    EC PostprocessOutput(const float* output, std::shared_ptr<Frame> frame);

private:

    bool initialized_ = false; 

    // Region ID
    RegionID region_id_;

    // Number of landmarks for the region
    int num_landmarks_;

    // Yolo model stride
    std::vector<int> strides_ = {8, 16, 32};

    // total number of yolo boxes
    int num_yolo_boxes;

    // confidence threshold for detected landmarks
    float confidence_threshold_ = 0.5f;

    // iou threshold for non-max suppression
    float iou_threshold_ = 0.5f;

    // Image size
    int input_width_ = 4608;

    // Output size
    int output_size_;

    // Landmark bounding box CSV file path (for post-processing)
    std::string csv_path_;

    // Logger for TensorRT
    Logger trt_logger_;

    // Inference buffer 
    InferenceBuffer buffers_;

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

static float ComputeIoU(const Landmark& a, const Landmark& b);

std::vector<Landmark> LDYoloNonMaxSuppression(
    const Eigen::MatrixXf& output_matrix,
    RegionID region_id,
    float conf_threshold,
    float iou_threshold);

int GetNumLandmarksFromCSV(const std::string& csv_path);

#endif // RUNTIMES_HPP