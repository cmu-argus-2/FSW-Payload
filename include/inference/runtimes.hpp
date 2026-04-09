#ifndef RUNTIMES_HPP
#define RUNTIMES_HPP

#include <string>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <eigen3/Eigen/Dense>

#include "spdlog/spdlog.h"

#include "inference/types.hpp"
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

class RCNet
{

public:

    RCNet();
    ~RCNet();

    // getters
    bool IsInitialized() const { return initialized_; }
    int GetBatchSize() const { return batch_size_; }
    int GetInputChannels() const { return input_channels_; }
    int GetInputHeight() const { return input_height_; }
    int GetInputWidth() const { return input_width_; }
    int GetNumClasses() const { return num_classes_; }

    EC LoadEngine(const std::string& engine_path);
    EC Free();
    EC Infer(const void* input_data, void* output);


private:

    bool initialized_ = false;

    // Input/output names
    // TODO: static or not? only used by runtime
    static constexpr const char* rc_input_name = "input";
    static constexpr const char* rc_output_name = "output";

    // Image size
    int input_width_;
    int input_height_;

    // Output size
    int output_size_;

    // Input channels
    int input_channels_;

    // Number of Region classes
    int num_classes_;

    // Batch Size
    int batch_size_;

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
    static int GetNumLandmarksFromCSV(const std::string& csv_path);
    int ComputeNumYoloBoxes() const;
    EC parseModelName(const std::string& name, LDNetConfig& ldnet_config);

    cv::Rect scaleBoxBackLetterbox(
        const cv::Rect& rBlob,
        const cv::Size& imgSize,
        const cv::Size& netSize);

    void yoloPostProcessing(
        cv::Mat outs,
        std::vector<int>& keep_classIds,
        std::vector<float>& keep_confidences,
        std::vector<cv::Rect2d>& keep_boxes);

    LDNet(RegionID region_id, std::string csv_path);
    ~LDNet();

    // getters
    bool IsInitialized() const { return initialized_; }
    
    // Explicitly release per-frame scratch allocations while keeping engine/context loaded
    EC ReleaseScratchBuffers();
    
    int GetNumLandmarks() const { return num_landmarks_; }
    int GetInputWidth() const { return input_width_; }
    int GetInputHeight() const { return input_height_; }
    int GetInputChannels() const { return input_channels_; }
    int GetBatchSize() const { return batch_size_; }
    int GetOutputSize() const { return output_size_; }
    float* GetOutputBuffer() { return output_buffer_.get(); }
    const float* GetOutputBuffer() const { return output_buffer_.get(); }
    int GetNumYoloBoxes() const { return num_yolo_boxes; }
    RegionID GetRegionID() const { return region_id_; }
    float GetConfidenceThreshold() const { return confidence_threshold_; }
    float GetIOUThreshold() const { return iou_threshold_; }
    bool IsDynamicSizeInput() const {return dynamic_size_input; }
    NET_QUANTIZATION GetNetQuantization() const {return weight_quant; }
    bool HasEmbeddedNMS() const {return embedded_nms; }
    bool HasClassAgnosticNMS() const {return class_agnostic_nms; }
    int GetTopKNMS() const {return topk_nms; }
    bool IsTRT() const { return is_trt; }

    // setters
    void SetInitialized(bool initialized) { initialized_ = initialized; }
    void SetLDNetConfig(LDNetConfig ldnet_config);

    EC Free();
    EC Infer(const void* input_data, void* output) ;
    EC Infer(cv::Mat input_data, std::vector<cv::Mat>& output);
    EC PostprocessOutput(cv::Mat outs, std::shared_ptr<Frame> frame);
    
    // Engine path is defined by the model parameters
    EC LoadEngine(const std::string& engine_path);
    // EC LoadEngine() {return LoadEngine(engine_path_);}; // Commented out for now
private:
    bool initialized_ = false;

    // True: TRT, False: ONNX
    bool is_trt = true;

    // Extra GPU bytes required beyond the natural estimate before any allocation.
    size_t gpu_reserve_bytes_ = 0;

    // Model with dynamic size
    bool dynamic_size_input = false;

    // Region ID
    RegionID region_id_;

    // Number of landmarks/classes for the region
    int num_landmarks_;

    // Yolo model stride
    const std::vector<int> strides_ = {8, 16, 32};

    // total number of yolo boxes
    int num_yolo_boxes;

    // confidence threshold for detected landmarks
    float confidence_threshold_ = 0.5f;

    // iou threshold for non-max suppression
    float iou_threshold_ = 0.45f;

    // Batch size
    const int batch_size_ = 1;

    // Input channels
    const int input_channels_ = 3;

    // Image size
    int input_width_ = 4608;
    int input_height_ = 2592;

    // Output size
    int output_size_;

    // Uses half-float precision
    NET_QUANTIZATION weight_quant = NET_QUANTIZATION::FP16;

    // If the model runs NMS
    bool embedded_nms = false;

    // If the NMS is class-agnostic
    bool class_agnostic_nms = false; // yolo default

    // Top K detections to keep in NMS
    int topk_nms = 300; // yolo default
    
    // Yolo model version
    std::string model_name_ = "yolov8";

    // Pre-allocated CPU output buffer for raw-detection engines.
    // Avoids a large heap allocation (~24–180 MB) on every inference call.
    std::unique_ptr<float[]> output_buffer_;

    // Landmark bounding box CSV file path (for post-processing)
    const std::string csv_path_;

    // Landmark engine path
    const std::string engine_path_;
    
    // TensorRT Engine
    // Logger for TensorRT
    Logger trt_logger_;

    // Inference buffer (input always; output only in raw-detection mode)
    InferenceBuffer buffers_;
    
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

    // ONNX Engine
    std::unique_ptr<cv::dnn::Net> net_ = nullptr;

};
} // namespace Inference

#endif // RUNTIMES_HPP
