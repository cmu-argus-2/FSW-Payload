#include "inference/runtimes.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/dnn/dnn.hpp>
// #include <opencv2/dnn/types.hpp>

// TODO: consider whether to remove this
using namespace cv;
using namespace cv::dnn;

namespace Inference
{

using namespace nvinfer1;

// TODO: Remove, define in classes
// static constexpr const char* rc_input_name = "input";
// static constexpr const char* rc_output_name = "output";
static constexpr const char* ld_input_name = "images";
static constexpr const char* ld_output_name = "output0";

// TODO: Can the input be obtained from the TRT? If so, either (a) a check should be 
// added when loading the engine or (b) these values are assigned when the engine is loaded.
// Either way, there should be a check for these values when they are assigned.
RCNet::RCNet(): // default values
batch_size_(1),
input_channels_(3),
input_height_(224),
input_width_(224),
num_classes_(16)
{
}

RCNet::~RCNet()
{
    cudaStreamDestroy(stream_);

    buffers_.free();
    
    if (context_) 
    {
        context_.reset();
    }
    if (engine_) 
    {
        engine_.reset();
    }
    if (runtime_) 
    {
        runtime_.reset();
    }
}

EC RCNet::LoadEngine(const std::string& engine_path)
{
    // TODO: Absolutely need to add retry and recovery logic
    
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) 
    {
        spdlog::error("Error: Failed to open engine file {}", engine_path);
        return EC::NN_FAILED_TO_OPEN_ENGINE_FILE;
    }
    file.seekg(0, std::ios::end); // Move > end of file
    size_t size = file.tellg(); 
    file.seekg(0, std::ios::beg); // Move >  beginning of the file
    std::vector<char> engine_data(size); 
    file.read(engine_data.data(), size);

    // Create runtime and deserialize engine
    runtime_ = std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(trt_logger_));
    if (!runtime_) {
        spdlog::error("Error: Failed to create runtime");
        LogError(EC::NN_FAILED_TO_CREATE_RUNTIME);
        return EC::NN_FAILED_TO_CREATE_RUNTIME;
    }

    engine_ = std::unique_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) {
        spdlog::error("Error: Failed to create CUDA engine");
        LogError(EC::NN_FAILED_TO_CREATE_ENGINE);
        return EC::NN_FAILED_TO_CREATE_ENGINE;
    }

    // Create execution context
    context_ = std::unique_ptr<IExecutionContext>(engine_->createExecutionContext());
    if (!context_) 
    {
        spdlog::error("Error: Failed to create execution context");
        LogError(EC::NN_FAILED_TO_CREATE_EXECUTION_CONTEXT);
        return EC::NN_FAILED_TO_CREATE_EXECUTION_CONTEXT;
    }

    assert(engine_->getTensorDataType(rc_input_name) == nvinfer1::DataType::kFLOAT);
    assert(engine_->getTensorDataType(rc_output_name) == nvinfer1::DataType::kFLOAT);

    context_->setInputShape(rc_input_name, nvinfer1::Dims4{batch_size_, input_channels_, input_height_, input_width_});

    // Allocate GPU memory for input and output
    // TODO: input shouldn't be float
    buffers_.input_size = batch_size_ * input_channels_ * input_height_ * input_width_ * sizeof(float);
    buffers_.output_size = num_classes_ * sizeof(float);
    buffers_.allocate();

    // Bind input and output memory (ok for RC)
    context_->setTensorAddress(rc_input_name, buffers_.input_data);
    context_->setTensorAddress(rc_output_name, buffers_.output_data);

    // Creating CUDA stream for asynchronous execution
    cudaStreamCreate(&stream_); 

    initialized_ = true;

    return EC::OK;
}

EC RCNet::Free()
{
    if (context_) context_.reset();
    if (engine_) engine_.reset();
    if (runtime_) runtime_.reset();
    buffers_.free();
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    initialized_ = false;
    spdlog::info("RCNet freed.");
    return EC::OK;
}


EC RCNet::Infer(const void* input_data, void* output)
{
    
    if (!initialized_) 
    {
        spdlog::error("RCNet is not initialized. Call LoadEngine first.");
        return EC::NN_ENGINE_NOT_INITIALIZED;
    }

    if (!input_data || !output) 
    {
        spdlog::error("Input data or output buffer is null.");
        LogError(EC::NN_POINTER_NULL);
        return EC::NN_POINTER_NULL;
    }

    // copy input data to GPU
    // spdlog::info("Copying input data to GPU memory.");
    cudaError_t err = cudaMemcpy(buffers_.input_data, input_data, buffers_.input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        spdlog::error("cudaMemcpyAsync (input) failed.");
        return EC::NN_CUDA_MEMCPY_FAILED;
    }

    // Run asynchronous inference with enqueueV3 (async by design)
    bool status = context_->enqueueV3(stream_); // could use the default stream with enqueueV3(0) adn then no need to sync
    if (!status)
    {
        spdlog::error("Failed to enqueue inference.");
        LogError(EC::NN_INFERENCE_FAILED);
        return EC::NN_INFERENCE_FAILED;
    }

    err = cudaMemcpyAsync(output, buffers_.output_data, buffers_.output_size, cudaMemcpyDeviceToHost, stream_); // Copy output back to CPU
    if (err != cudaSuccess) 
    {
        spdlog::error("cudaMemcpyAsync output failed.");
        LogError(EC::NN_CUDA_MEMCPY_FAILED);
        return EC::NN_CUDA_MEMCPY_FAILED;
    }
    // spdlog::info("Inference enqueued successfully. Waiting for completion...");
    cudaStreamSynchronize(stream_);  
    return EC::OK;
}

// LDNet methods

LDNet::LDNet(RegionID region_id, std::string csv_path)
: region_id_(region_id), csv_path_(csv_path), 
num_landmarks_(GetNumLandmarksFromCSV(csv_path)), // 
input_width_(4608), input_height_(2592), input_channels_(3), batch_size_(1)
{
    num_yolo_boxes = ComputeNumYoloBoxes();
    output_size_ = batch_size_ * (num_landmarks_ + 4) * num_yolo_boxes; // Update output size with the computed number of YOLO boxes
    spdlog::info("LDNet created for region {}. CSV path: {}. Num landmarks: {}. Num YOLO boxes: {}. Input Width: {}. Input Height: {}", 
                GetRegionString(region_id), csv_path, num_landmarks_, num_yolo_boxes, input_width_, input_height_);
}


int LDNet::GetNumLandmarksFromCSV(const std::string& csv_path)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) 
    {
        spdlog::error("Failed to open CSV file: {}", csv_path);
        return -1; // or throw an exception
    }

    std::string line;
    int num_landmarks = 0;
    bool first_line = true;
    while (std::getline(file, line)) 
    {
        if (first_line) 
        {
            first_line = false; // skip header
            continue;
        }
        if (!line.empty()) 
        {
            num_landmarks++;
        }
    }
    file.close();
    return num_landmarks;
}

int LDNet::ComputeNumYoloBoxes() const
{
    int total_boxes = 0;
    for (const auto& stride : strides_)
    {
        total_boxes += std::floor(GetInputWidth() / stride) * std::floor(GetInputHeight() / stride);
    
    }
    return total_boxes;
}

void LDNet::SetLDNetConfig(LDNetConfig ldnet_config)
{
    weight_quant = ldnet_config.weight_quant;
    input_width_  = ldnet_config.input_width;
    input_height_ = ldnet_config.input_height;
    embedded_nms  = ldnet_config.embedded_nms;
    // input_width  = ldnet_config.use_trt;
}

// TRTLD Net methods

TRTLDNet::TRTLDNet(RegionID region_id, std::string csv_path)
: LDNet(region_id, csv_path)
{
}

TRTLDNet::~TRTLDNet()
{
    cudaStreamDestroy(stream_);

    buffers_.free();
    
    if (context_) 
    {
        context_.reset();
    }
    if (engine_) 
    {
        engine_.reset();
    }
    if (runtime_) 
    {
        runtime_.reset();
    }
}

EC TRTLDNet::LoadEngine(const std::string& engine_path)
{
    // TODO: Absolutely need to add retry and recovery logic
    
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) 
    {
        spdlog::error("Error: Failed to open engine file {}", engine_path);
        return EC::NN_FAILED_TO_OPEN_ENGINE_FILE;
    }
    file.seekg(0, std::ios::end); // Move > end of file
    size_t size = file.tellg(); 
    file.seekg(0, std::ios::beg); // Move >  beginning of the file
    std::vector<char> engine_data(size); 
    file.read(engine_data.data(), size);

    // Create runtime and deserialize engine
    runtime_ = std::unique_ptr<IRuntime>(nvinfer1::createInferRuntime(trt_logger_));
    if (!runtime_) {
        spdlog::error("Error: Failed to create runtime");
        LogError(EC::NN_FAILED_TO_CREATE_RUNTIME);
        return EC::NN_FAILED_TO_CREATE_RUNTIME;
    }

    engine_ = std::unique_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) {
        spdlog::error("Error: Failed to create CUDA engine");
        LogError(EC::NN_FAILED_TO_CREATE_ENGINE);
        return EC::NN_FAILED_TO_CREATE_ENGINE;
    }

    // Create execution context
    context_ = std::unique_ptr<IExecutionContext>(engine_->createExecutionContext());
    if (!context_) 
    {
        spdlog::error("Error: Failed to create execution context");
        LogError(EC::NN_FAILED_TO_CREATE_EXECUTION_CONTEXT);
        return EC::NN_FAILED_TO_CREATE_EXECUTION_CONTEXT;
    }

    assert(engine_->getTensorDataType(ld_input_name) == nvinfer1::DataType::kFLOAT);
    assert(engine_->getTensorDataType(ld_output_name) == nvinfer1::DataType::kFLOAT);

    int batch_size = GetBatchSize();
    int input_channels = GetInputChannels();
    int input_height = GetInputHeight();
    int input_width = GetInputWidth();
    int num_yolo_boxes = GetNumYoloBoxes();

    // TODO: Check that these values are valid and match the engine's expected input dimensions.

    context_->setInputShape(ld_input_name, nvinfer1::Dims4{batch_size, input_channels, input_height, input_width});

    // Allocate GPU memory for input and output
    buffers_.input_size = batch_size * input_channels * input_height * input_width * sizeof(float);
    buffers_.output_size = batch_size * (GetNumLandmarks() + 4) * num_yolo_boxes * sizeof(float);
    buffers_.allocate();

    // Bind input and output memory (ok for LD?)
    context_->setTensorAddress(ld_input_name, buffers_.input_data);
    context_->setTensorAddress(ld_output_name, buffers_.output_data);

    // Creating CUDA stream for asynchronous execution
    cudaStreamCreate(&stream_); 

    SetInitialized(true);

    return EC::OK;
}

EC TRTLDNet::Free()
{
    if (context_) context_.reset();
    if (engine_) engine_.reset();
    if (runtime_) runtime_.reset();
    buffers_.free();
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    SetInitialized(false);
    spdlog::info("LDNet freed.");
    return EC::OK;
}

EC TRTLDNet::Infer(const void* input_data, void* output)
{
    // const void* input_data
    if (!IsInitialized()) 
    {
        spdlog::error("LDNet is not initialized. Call LoadEngine first.");
        return EC::NN_ENGINE_NOT_INITIALIZED;
    }

    if (!input_data || !output) 
    {
        spdlog::error("Input data or output buffer is null.");
        LogError(EC::NN_POINTER_NULL);
        return EC::NN_POINTER_NULL;
    }

    // copy input data to GPU
   // spdlog::info("Copying input data to GPU memory.");
    cudaError_t err = cudaMemcpy(buffers_.input_data, input_data, buffers_.input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) 
    {
        spdlog::error("cudaMemcpyAsync (input) failed.");
        return EC::NN_CUDA_MEMCPY_FAILED;
    }

    // Run asynchronous inference with enqueueV3 (async by design)
    bool status = context_->enqueueV3(stream_); // could use the default stream with enqueueV3(0) adn then no need to sync
    if (!status)
    {
        spdlog::error("Failed to enqueue inference.");
        LogError(EC::NN_INFERENCE_FAILED);
        return EC::NN_INFERENCE_FAILED;
    }

    err = cudaMemcpyAsync(output, buffers_.output_data, buffers_.output_size, cudaMemcpyDeviceToHost, stream_); // Copy output back to CPU
    if (err != cudaSuccess) 
    {
        spdlog::error("cudaMemcpyAsync output failed.");
        LogError(EC::NN_CUDA_MEMCPY_FAILED);
        return EC::NN_CUDA_MEMCPY_FAILED;
    }
    // spdlog::info("Inference enqueued successfully. Waiting for completion...");
    cudaStreamSynchronize(stream_);  
    return EC::OK;
}

EC TRTLDNet::PostprocessOutput(float* ld_output, std::shared_ptr<Frame> frame)
{
    std::vector<int> keep_classIds;
    std::vector<float> keep_confidences;
    std::vector<Rect2d> keep_boxes;
    std::vector<Rect> boxes;

    int sizes[3] = {1, GetNumLandmarks() + 4, GetNumYoloBoxes()};

    cv::Mat output_matrix(3, sizes, CV_32F, ld_output);

    // Non-max suppression
    LDNet::yoloPostProcessing(output_matrix, keep_classIds, keep_confidences, keep_boxes);

    for (auto box : keep_boxes)
    {
        boxes.push_back(Rect(cvFloor(box.x), cvFloor(box.y), cvFloor(box.width - box.x), cvFloor(box.height - box.y)));
    }

    for (auto& b : boxes) {
        b = scaleBoxBackLetterbox(b, frame->GetImgSize(), 
                                    cv::Size(GetInputWidth(), GetInputHeight()));
    }

    // Populate landmarks
    Rect2d box;
    float x_center, y_center, width, height;
    uint16_t class_id;
    for (int i = 0; i < boxes.size(); ++i)
    {
        Rect box = boxes[i];

        // TODO: Revise variable datatype
        height = static_cast<float>(box.height);
        width = static_cast<float>(box.width);
        x_center = static_cast<float>(box.x) + width / 2.0f;
        y_center = static_cast<float>(box.y) + height / 2.0f;

        class_id = static_cast<uint16_t>(keep_classIds[i]);
        Landmark landmark(x_center, y_center, class_id, GetRegionID(),
                            height, width, keep_confidences[i]);
        // Landmark landmark(keep_boxes[i], keep_classIds[i], keep_confidences[i]);
        frame->AddLandmark(landmark);
    }

    return EC::OK;
}

// ONNXLDNet methods
void LDNet::yoloPostProcessing(
    Mat outs,
    std::vector<int>& keep_classIds,
    std::vector<float>& keep_confidences,
    std::vector<Rect2d>& keep_boxes)
{
    float conf_threshold = GetConfidenceThreshold();
    float iou_threshold = GetIOUThreshold();
    int nc = GetNumLandmarks();

    // Retrieve
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> boxes;

    if (model_name_ == "yolov8" || model_name_ == "yolov10" ||
        model_name_ == "yolov9")
    {
        // TODO: try to not use this 
        cv::transposeND(outs, {0, 2, 1}, outs);
    }

    // assert if last dim is nc+5 or nc+4
    // TODO: These CheckEQ can be used in testing but not in the code
    spdlog::info("Output shape: [{}, {}, {}]", outs.size[0], outs.size[1], outs.size[2]);
    spdlog::info("Expected output shape: [1, #anchors, nc+5 or nc+4] where nc is {}", nc);
    CV_CheckEQ(outs.dims, 3, "Invalid output shape. The shape should be [1, #anchors, nc+5 or nc+4]");
    CV_CheckEQ((outs.size[2] == nc + 5 || outs.size[2] == nc + 4), true, "Invalid output shape: ");


    auto preds = outs.reshape(1, outs.size[1]); // [1, 8400, 85] -> [8400, 85]
    for (int i = 0; i < preds.rows; ++i)
    {
        // filter out non object
        float obj_conf = (model_name_ == "yolov8" || model_name_ == "yolonas" ||
                            model_name_ == "yolov9" || model_name_ == "yolov10") ? 1.0f : preds.at<float>(i, 4) ;
        if (obj_conf < conf_threshold)
            continue;

        Mat scores = preds.row(i).colRange((model_name_ == "yolov8" || model_name_ == "yolonas" || model_name_ == "yolov9" || model_name_ == "yolov10") ? 4 : 5, preds.cols);
        double conf;
        Point maxLoc;
        minMaxLoc(scores, 0, &conf, 0, &maxLoc);

        conf = (model_name_ == "yolov8" || model_name_ == "yolonas" || model_name_ == "yolov9" || model_name_ == "yolov10") ? conf : conf * obj_conf;
        if (conf < conf_threshold)
            continue;

        // get bbox coords
        float* det = preds.ptr<float>(i);
        double cx = det[0];
        double cy = det[1];
        double w = det[2];
        double h = det[3];

        // [x1, y1, x2, y2]
        if (model_name_ == "yolonas" || model_name_ == "yolov10"){
            boxes.push_back(Rect2d(cx, cy, w, h));
        } else {
            boxes.push_back(Rect2d(cx - 0.5 * w, cy - 0.5 * h,
                                    cx + 0.5 * w, cy + 0.5 * h));
        }
        classIds.push_back(maxLoc.x);
        confidences.push_back(static_cast<float>(conf));
    }

    spdlog::info("Boxes: {}, Confidences: {}", boxes.size(), confidences.size());

    // NMS
    std::vector<int> keep_idx;
    NMSBoxesBatched(boxes, confidences, classIds, conf_threshold, iou_threshold, keep_idx);

    for (auto i : keep_idx)
    {
        keep_classIds.push_back(classIds[i]);
        keep_confidences.push_back(confidences[i]);
        keep_boxes.push_back(boxes[i]);
    }
}

cv::Rect LDNet::scaleBoxBackLetterbox(
    const cv::Rect& rBlob,
    const cv::Size& imgSize,
    const cv::Size& netSize)
{
    const float gain = std::min(netSize.width / (float)imgSize.width,
                                netSize.height / (float)imgSize.height);
    const float padX = (netSize.width  - imgSize.width  * gain) * 0.5f;
    const float padY = (netSize.height - imgSize.height * gain) * 0.5f;

    float x = (rBlob.x - padX) / gain;
    float y = (rBlob.y - padY) / gain;
    float w = rBlob.width  / gain;
    float h = rBlob.height / gain;

    cv::Rect r((int)std::round(x), (int)std::round(y),
               (int)std::round(w), (int)std::round(h));
    return r & cv::Rect(0, 0, imgSize.width, imgSize.height);
}

ONNXLDNet::ONNXLDNet(RegionID region_id, std::string csv_path)
: LDNet(region_id, csv_path)
{
}

ONNXLDNet::~ONNXLDNet()
{
    // deallocate LDNet
}

EC ONNXLDNet::LoadEngine(const std::string& engine_path)
{
    SetInitialized(true);
    // TODO: Error handling
    net_ = std::make_unique<cv::dnn::Net>(cv::dnn::readNet(engine_path));
    net_->setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net_->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    return EC::OK;
}

EC ONNXLDNet::Free()
{
    net_.reset();
    SetInitialized(false);
    spdlog::info("LDNet freed.");
    return EC::OK;
}

EC ONNXLDNet::Infer(cv::Mat input_data, std::vector<cv::Mat>& output)
{
    
    if (!IsInitialized()) 
    {
        spdlog::error("LDNet is not initialized. Call LoadEngine first.");
        return EC::NN_ENGINE_NOT_INITIALIZED;
    }

   net_->setInput(input_data);
   net_->forward(output, net_->getUnconnectedOutLayersNames());
   spdlog::info("Forward pass completed, outputs: [{}, {}, {}]", output[0].size[0], output[0].size[1], output[0].size[2]);
   
   return EC::OK;
}

EC ONNXLDNet::PostprocessOutput(std::vector<cv::Mat> outs, std::shared_ptr<Frame> frame)
{
    // TODO
    std::vector<int> keep_classIds;
    std::vector<float> keep_confidences;
    std::vector<Rect2d> keep_boxes;
    std::vector<Rect> boxes;

    // Non-max suppression
    yoloPostProcessing(outs[0], keep_classIds, keep_confidences, keep_boxes);

    for (auto box : keep_boxes)
    {
        boxes.push_back(Rect(cvFloor(box.x), cvFloor(box.y), cvFloor(box.width - box.x), cvFloor(box.height - box.y)));
    }

    for (auto& b : boxes) {
        b = scaleBoxBackLetterbox(b, frame->GetImgSize(), 
                                    cv::Size(GetInputWidth(), GetInputHeight()));
    }

    // Populate landmarks
    Rect2d box;
    float x_center, y_center, width, height;
    uint16_t class_id;
    for (int i = 0; i < boxes.size(); ++i)
    {
        Rect box = boxes[i];

        // TODO: Revise variable datatype
        height = static_cast<float>(box.height);
        width = static_cast<float>(box.width);
        x_center = static_cast<float>(box.x) + width / 2.0f;
        y_center = static_cast<float>(box.y) + height / 2.0f;

        class_id = static_cast<uint16_t>(keep_classIds[i]);
        Landmark landmark(x_center, y_center, class_id, GetRegionID(),
                            height, width, keep_confidences[i]);
        // Landmark landmark(keep_boxes[i], keep_classIds[i], keep_confidences[i]);
        frame->AddLandmark(landmark);
    }
    

    return EC::OK;
}


}