#include "inference/runtimes.hpp"

#include <vector>
#include <fstream>
#include <iostream>
namespace Inference
{

using namespace nvinfer1;

static constexpr const char* rc_input_name = "input";
static constexpr const char* rc_output_name = "output";
static constexpr const char* ld_input_name = "images";
static constexpr const char* ld_output_name = "output0";

RCNet::RCNet()
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

    context_->setInputShape(rc_input_name, nvinfer1::Dims4{BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH});

    // Allocate GPU memory for input and output
    buffers_.input_size = BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
    buffers_.output_size = RC_NUM_CLASSES * sizeof(float);
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

// LD Net methods

LDNet::LDNet(RegionID region_id, std::string csv_path)
: region_id_(region_id), csv_path_(csv_path)
{

    num_landmarks_ = GetNumLandmarksFromCSV(csv_path_);
    num_yolo_boxes = 0;
    for (const auto& stride : strides_)
    {
        num_yolo_boxes += (input_width_ / stride) * (input_width_ / stride);
    }
    output_size_ = (num_landmarks_ + 4) * num_yolo_boxes; // 4 for bounding box coordinates
}

LDNet::~LDNet()
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

EC LDNet::LoadEngine(const std::string& engine_path)
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

    context_->setInputShape(ld_input_name, nvinfer1::Dims4{BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH});

    // Allocate GPU memory for input and output
    buffers_.input_size = BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
    buffers_.output_size = BATCH_SIZE * (num_landmarks_ + 4) * num_yolo_boxes * sizeof(float);
    buffers_.allocate();

    // Bind input and output memory (ok for LD?)
    context_->setTensorAddress(ld_input_name, buffers_.input_data);
    context_->setTensorAddress(ld_output_name, buffers_.output_data);

    // Creating CUDA stream for asynchronous execution
    cudaStreamCreate(&stream_); 

    initialized_ = true;

    return EC::OK;
}

EC LDNet::Free()
{
    if (context_) context_.reset();
    if (engine_) engine_.reset();
    if (runtime_) runtime_.reset();
    buffers_.free();
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    initialized_ = false;
    spdlog::info("LDNet freed.");
    return EC::OK;
}

EC LDNet::Infer(const void* input_data, void* output)
{
    
    if (!initialized_) 
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

EC LDNet::PostprocessOutput(const float* ld_output, std::shared_ptr<Frame> frame)
{
    // store ld_output in a matrix of size [num_landmarks + 4, num_boxes]
    int total_output_size = (num_landmarks_ + 4) * num_yolo_boxes;
    // Eigen::MatrixXf output_matrix(num_landmarks_ + 4, num_yolo_boxes);
    // for (int i = 0; i < total_output_size; ++i) {
    //     if (i % 10000 == 0) {
    //         spdlog::info("Processing output element {}/{}", i, total_output_size);
    //     }
    //     output_matrix(i / (num_yolo_boxes), i % (num_yolo_boxes)) = ld_output[i];
    // }

    using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMat> output_matrix(ld_output, num_landmarks_ + 4, num_yolo_boxes);

    // Non-max suppression
    std::vector<Landmark> landmarks = LDYoloNonMaxSuppression(output_matrix, region_id_, confidence_threshold_, iou_threshold_);
    
    
    // Populate landmarks
    for (const auto& landmark : landmarks)
    {
        frame->AddLandmark(landmark);
    }
    

    return EC::OK;
}

}


static float ComputeIoU(const Landmark& a, const Landmark& b)
{
    const float ax1 = a.x - 0.5f * a.width;
    const float ay1 = a.y - 0.5f * a.height;
    const float ax2 = a.x + 0.5f * a.width;
    const float ay2 = a.y + 0.5f * a.height;

    const float bx1 = b.x - 0.5f * b.width;
    const float by1 = b.y - 0.5f * b.height;
    const float bx2 = b.x + 0.5f * b.width;
    const float by2 = b.y + 0.5f * b.height;

    const float inter_x1 = std::max(ax1, bx1);
    const float inter_y1 = std::max(ay1, by1);
    const float inter_x2 = std::min(ax2, bx2);
    const float inter_y2 = std::min(ay2, by2);

    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    const float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
    const float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
    const float union_area = area_a + area_b - inter_area;

    return inter_area / (union_area + 1e-6f);
}

// If member: std::vector<Landmark> LDNet::LDYoloNonMaxSuppression(...)
std::vector<Landmark> LDYoloNonMaxSuppression(
    const Eigen::MatrixXf& output_matrix,
    RegionID region_id,
    float conf_threshold,
    float iou_threshold)
{
    std::vector<Landmark> candidates;
    if (output_matrix.rows() < 5 || output_matrix.cols() == 0) return candidates;

    const int num_classes = output_matrix.rows() - 4;
    const int num_boxes = output_matrix.cols();
    candidates.reserve(num_boxes);

    // Decode + confidence filter
    for (int i = 0; i < num_boxes; ++i)
    {
        int best_class = 0;
        float best_conf = output_matrix(4, i);
        for (int c = 1; c < num_classes; ++c)
        {
            const float score = output_matrix(4 + c, i);
            if (score > best_conf)
            {
                best_conf = score;
                best_class = c;
            }
        }
        // TODO: investigate why we are getting a few NaN confidence scores
        if (std::isnan(best_conf) || best_conf < 0.0f)
        {
            spdlog::warn("Box {} has invalid confidence score: {:.3f}. Skipping.", i, best_conf);
            continue;
        }

        if (best_conf < conf_threshold) continue;

        const float x = output_matrix(0, i);
        const float y = output_matrix(1, i);
        const float w = output_matrix(2, i);
        const float h = output_matrix(3, i);

        candidates.emplace_back(
            x, y,
            static_cast<uint16_t>(best_class),
            region_id,
            h, w,                  // Landmark(height, width)
            best_conf);
    }

    if (candidates.empty()) return candidates;

    // Shrink candidates vector to remove unused capacity
    candidates.shrink_to_fit();

    // Sort by confidence desc
    std::sort(candidates.begin(), candidates.end(),
              [](const Landmark& a, const Landmark& b) {
                  return a.confidence > b.confidence;
              });

    // Class-aware NMS
    std::vector<Landmark> kept;
    std::vector<bool> suppressed(candidates.size(), false);
    kept.reserve(candidates.size());

    for (size_t i = 0; i < candidates.size(); ++i)
    {
        if (suppressed[i]) continue;
        kept.push_back(candidates[i]);

        for (size_t j = i + 1; j < candidates.size(); ++j)
        {
            if (suppressed[j]) continue;
            if (candidates[j].class_id != candidates[i].class_id) continue;
            // keep only the highest confidence class detetion
            suppressed[j] = true;
            // use iou to decide
            // if (ComputeIoU(candidates[i], candidates[j]) > iou_threshold)
            // {
            //     suppressed[j] = true;
            // }
        }
    }

    return kept;
}

int GetNumLandmarksFromCSV(const std::string& csv_path)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) 
    {
        spdlog::error("Failed to open CSV file: {}", csv_path);
        return -1; // or throw an exception
    }

    std::string line;
    int num_landmarks = 0;
    while (std::getline(file, line)) 
    {
        if (!line.empty()) 
        {
            num_landmarks++;
        }
    }
    file.close();
    return num_landmarks;
}