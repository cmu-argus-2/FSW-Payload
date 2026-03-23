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

// TODO: Remove, define in classes
// static constexpr int BATCH_SIZE = 1; // Batch size for inference
// static constexpr int INPUT_CHANNELS = 3; // Number of input channels (RGB)
// static constexpr int INPUT_HEIGHT = 224; // Input height
// static constexpr int INPUT_WIDTH = 224; // Input width
// static constexpr int RC_NUM_CLASSES = 16; // Number of classes in the output (THIS CHANGES WITH MODEL, WILL BREAK OTHERWISE)

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

enum class NET_QUANTIZATION : uint8_t 
{
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
};

inline std::string GetQuantString(NET_QUANTIZATION quant) {
    switch (quant) {
        case NET_QUANTIZATION::FP32: return "fp32";
        case NET_QUANTIZATION::FP16: return "fp16";
        case NET_QUANTIZATION::INT8: return "int8";
        default: return "";
    }
}

// Parameters used to define the model file name (knowing the region)
struct LDNetConfig {
    NET_QUANTIZATION weight_quant;
    int input_width;
    int input_height;
    bool embedded_nms;
    bool use_trt;

    std::string GetFileNameAppendix() {

        std::string fp16_string = GetQuantString(weight_quant);

        std::string nms_string = "";
        if (embedded_nms) nms_string = "_nms";

        std::string file_ext = "onnx";
        if (use_trt) file_ext = "trt";

        return "_weights_" + fp16_string + "_sz_" + std::to_string(input_width) + nms_string + "." + file_ext;
    }
};


class LDNet
{
public:
    static int GetNumLandmarksFromCSV(const std::string& csv_path);
    int ComputeNumYoloBoxes() const;

    LDNet(RegionID region_id, std::string csv_path);
    virtual ~LDNet() = default;

    // getters
    bool IsInitialized() const { return initialized_; }
    int GetNumLandmarks() const { return num_landmarks_; }
    int GetInputWidth() const { return input_width_; }
    int GetInputHeight() const { return input_height_; }
    int GetInputChannels() const { return input_channels_; }
    int GetBatchSize() const { return batch_size_; }
    int GetOutputSize() const { return output_size_; }
    int GetNumYoloBoxes() const { return num_yolo_boxes; }
    RegionID GetRegionID() const { return region_id_; }
    float GetConfidenceThreshold() const { return confidence_threshold_; }
    float GetIOUThreshold() const { return iou_threshold_; }
    bool IsDynamicSizeInput() const {return dynamic_size_input; }
    NET_QUANTIZATION GetNetQuantization() const {return weight_quant; }
    bool HasEmbeddedNMS() const {return embedded_nms; }
    bool HasClassAgnosticNMS() const {return class_agnostic_nms; }
    int GetTopKNMS() const {return topk_nms; }

    // setters
    void SetInitialized(bool initialized) { initialized_ = initialized; }
    void SetLDNetConfig(LDNetConfig ldnet_config);

    

    virtual EC Free() = 0;
    virtual EC Infer(const void* input_data, void* output) = 0;
    virtual EC Infer(cv::Mat input_data, std::vector<cv::Mat>& output) = 0;
    // TODO: Once OpenCV with CUDA support is available, change PostprocessOutput to take cv::Mat outs
    virtual EC PostprocessOutput(const float* output, std::shared_ptr<Frame> frame) = 0;
    virtual EC PostprocessOutput(std::vector<cv::Mat> outs, std::shared_ptr<Frame> frame) = 0;

    // Engine path is defined by the model parameters
    virtual EC LoadEngine(const std::string& engine_path) = 0;
    EC LoadEngine() {return LoadEngine(engine_path_);};
private:
    bool initialized_ = false;

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
    bool topk_nms = 300; // yolo default

    // Landmark bounding box CSV file path (for post-processing)
    const std::string csv_path_;

    // Landmark engine path
    const std::string engine_path_;

};

class TRTLDNet: public LDNet
{

public:
    // static methods
    // TODO: method shouldn't be static
    static std::vector<Landmark> LDYoloNonMaxSuppression(
        const Eigen::MatrixXf& output_matrix,
        RegionID region_id,
        float conf_threshold,
        float iou_threshold);

    TRTLDNet(RegionID region_id, std::string csv_path);
    ~TRTLDNet();

    EC LoadEngine(const std::string& engine_path);

    EC Free();
    EC Infer(const void* input_data, void* output);
    EC Infer(cv::Mat input_data, std::vector<cv::Mat>& output) { return EC::OK; } // Not used for TRT, only the raw pointer version is used;
    EC PostprocessOutput(std::vector<cv::Mat> outs, std::shared_ptr<Frame> frame)  { return EC::OK; }
    // TODO: Once OpenCV with CUDA support is available, change PostprocessOutput to take cv::Mat outs
    EC PostprocessOutput(const float* output, std::shared_ptr<Frame> frame);

private:
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

class ONNXLDNet: public LDNet
{

public:
    cv::Rect scaleBoxBackLetterbox(
        const cv::Rect& rBlob,
        const cv::Size& imgSize,
        const cv::Size& netSize);
        
    // TODO: Name consistent between ONNX and TRT. Have a virtual method in LDNet
    void yoloPostProcessing(
        std::vector<cv::Mat>& outs,
        std::vector<int>& keep_classIds,
        std::vector<float>& keep_confidences,
        std::vector<cv::Rect2d>& keep_boxes);

    ONNXLDNet(RegionID region_id, std::string csv_path);
    ~ONNXLDNet();

    EC LoadEngine(const std::string& engine_path);

    EC Free();
    EC Infer(const void* input_data, void* output) { return EC::OK; } // Not used for ONNX, only the cv::Mat version is used;
    EC Infer(cv::Mat input_data, std::vector<cv::Mat>& output);
    EC PostprocessOutput(std::vector<cv::Mat> outs, std::shared_ptr<Frame> frame);
    // TODO: Once OpenCV with CUDA support is available, change PostprocessOutput to take cv::Mat outs
    EC PostprocessOutput(const float* output, std::shared_ptr<Frame> frame) { return EC::OK; }

private:
    std::unique_ptr<cv::dnn::Net> net_ = nullptr;
    std::string model_name_ = "yolov8"; // default to yolov8, can be set based on region or csv in the future
};

} // namespace Inference

#endif // RUNTIMES_HPP