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

// NOTE: This specifies the size of the networks, must change with model changes
static constexpr int BATCH_SIZE = 1;       // Batch size for inference
static constexpr int INPUT_CHANNELS = 3;   // Number of input channels (RGB)
static constexpr int RC_INPUT_HEIGHT = 224; // RCNet input height (EfficientNet-B0)
static constexpr int RC_INPUT_WIDTH  = 224; // RCNet input width  (EfficientNet-B0)
static constexpr int RC_NUM_CLASSES = 16;  // Number of classes in the output (THIS CHANGES WITH MODEL, WILL BREAK OTHERWISE)
static constexpr int LD_INPUT_CHANNELS = 3; // Number of input channels (RGB)
static constexpr int LD_INPUT_HEIGHT = 4608; // LDNet input height (square)
static constexpr int LD_INPUT_WIDTH  = 4608; // LDNet input width  (square)
static constexpr int MAX_DETECTIONS = 300; // Max detections from EfficientNMS_TRT plugin

class RCNet
{

public:

    RCNet();
    ~RCNet();

    EC LoadEngine(const std::string& engine_path);
    EC Free();
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
    bool IsNMSEngine()   const { return nms_in_engine_; }
    
    EC Free();
    // Explicitly release per-frame scratch allocations while keeping engine/context loaded
    EC ReleaseScratchBuffers();
    
    EC Infer(const void* input_data, void* output);
    int GetNumLandmarks() const { return num_landmarks_; }
    
    // Returns 0 for engines with NMS built-in (no CPU output buffer needed — results are in NMS member buffers)
    // NOTE: Currently, TRT issues have prevented this
    int GetOutputSize() const { return nms_in_engine_ ? 0 : output_size_; }
    float* GetOutputBuffer() const { return output_buffer_.get(); }
    
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

    // Output size (raw detection mode only; 0 for NMS engines)
    int output_size_;

    // Pre-allocated CPU output buffer for raw-detection engines.
    // Avoids a large heap allocation (~24–180 MB) on every inference call.
    std::unique_ptr<float[]> output_buffer_;

    // Landmark bounding box CSV file path (for post-processing)
    std::string csv_path_;

    // Logger for TensorRT
    Logger trt_logger_;

    // Inference buffer (input always; output only in raw-detection mode)
    InferenceBuffer buffers_;

    // TODO EfficientNMS DOES NOT CURRENTLY WORK
    // This would dramatically help ram usage, but the dependencies with TRT conversion are a mess
    
    // --- EfficientNMS_TRT mode ---
    // Set to true when the loaded engine has 5 IO tensors (1 input + 4 NMS outputs).
    bool nms_in_engine_ = false;

    // GPU output buffers only allocated when nms_in_engine_ = true
    void* nms_num_dets_gpu_    = nullptr;  // int32 [1,1]
    void* nms_det_boxes_gpu_   = nullptr;  // float [1,MAX_DETECTIONS,4] XYXY pixels
    void* nms_det_scores_gpu_  = nullptr;  // float [1,MAX_DETECTIONS]
    void* nms_det_classes_gpu_ = nullptr;  // int32 [1,MAX_DETECTIONS]

    // CPU-side copies of NMS outputs
    int32_t nms_num_dets_cpu_[1]                   = {};
    float   nms_det_boxes_cpu_[MAX_DETECTIONS * 4] = {};
    float   nms_det_scores_cpu_[MAX_DETECTIONS]    = {};
    int32_t nms_det_classes_cpu_[MAX_DETECTIONS]   = {};
    
    // The scratch buffer tracks if the memory is allocated currently. 
    // Allocated in EnsureScratchBuffers(), released in ReleaseScratchBuffers(). Aggressively Freed
    bool scratch_buffers_allocated_ = false;
    // ---

    // Lazily allocate/reallocate and bind CUDA + host scratch buffers before Infer().
    // THIS SHOULD BE VALIDATED
    EC EnsureScratchBuffers();

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

// Converting the output matrix to be row-major prevents unnecessary copy -- saves ~900MB RAM usage
// NOTE: For future reference, Eigen::Map should be RowMajor!!
using LDOutputMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using LDOutputMatrixRef = Eigen::Ref<const LDOutputMatrix>;

std::vector<Landmark> LDYoloNonMaxSuppression(
    const LDOutputMatrixRef& output_matrix,
    RegionID region_id,
    float conf_threshold,
    float iou_threshold);

int GetNumLandmarksFromCSV(const std::string& csv_path);

#endif // RUNTIMES_HPP