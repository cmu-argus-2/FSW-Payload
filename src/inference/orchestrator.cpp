#include "inference/orchestrator.hpp"
#include "spdlog/spdlog.h"

#include <opencv2/opencv.hpp>
// #include <opencv2/core/cuda.hpp>
// #include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudaimgproc.hpp>

namespace Inference
{

Orchestrator::Orchestrator()
: current_frame_(nullptr)
{

}

void Orchestrator::Initialize(const std::string& rc_engine_path)
{
    // Initialize the RCNet runtime
    EC rc_net_status = rc_net_.LoadEngine(rc_engine_path);
    if (rc_net_status != EC::OK) 
    {
        spdlog::error("Failed to load RC Net engine.");
        return; // could have fallback in which we still run the LDs but that is out of scope for now
    }

    // TODO: Initialize the LDs runtimes

}

void Orchestrator::GrabNewImage(std::shared_ptr<Frame> frame)
{
    if (!frame) 
    {
        spdlog::error("Received null frame.");
        return;
    }

    original_frame_ = frame; 
    current_frame_ = std::make_shared<Frame>(*frame); // deep copy
    num_inference_performed_on_current_frame_ = 0; // Reset the inference count for the new frame
}

size_t Orchestrator::GetMemorySize(const nvinfer1::Dims& dims, size_t element_size)
{
    size_t size = element_size;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}

void Orchestrator::PreprocessImg(cv::Mat img, cv::Mat& out_chw_img)
{
    // TODO: optimize and remove copies. Leverage GPU

    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    // 1. Resize
    cv::resize(img, img, cv::Size(224, 224), 0, 0, cv::INTER_AREA); 
    // Note: cv::INTER_AREA is the best one I found closest to torchvision.transforms.Resize (which uses PIL resize's default)
    // Ideally, we would want to avoid that because it has *slight* pixel differences due to their implementation
    // spdlog::info("Image resized to 224x224, current shape: {}x{}x{}", img.rows, img.cols, img.channels());

    // 2. ToTensor (Convert to float and scale to [0, 1]) 
    // Convert from uint8 [0, 255] to float32 [0, 1]
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0 / 255.0);  // Correctly normalize
    // Convert HWC (224x224x3) to CHW (3x224x224)
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);  // Split into individual float32 channels

    // 3. Normalize per channel
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    // Apply normalization per channel
    for (int c = 0; c < 3; c++) 
    {
        channels[c] = (channels[c] - mean[c]) / std[c];  // Element-wise operation
    }

    // Stack back to CHW
    // spdlog::info("Stacking channels to CHW format, current shape: {}x{}x{}", channels[0].rows, channels[0].cols, channels.size());

    // spdlog::info("Size of out_chw_img before: {}x{}x{}", out_chw_img.rows, out_chw_img.cols, out_chw_img.channels());
    cv::vconcat(channels, out_chw_img);  // Stack channels vertically (CHW format)
    // spdlog::info("Size of out_chw_img after: {}x{}x{}", out_chw_img.rows, out_chw_img.cols, out_chw_img.channels());
    // TODO: preprocessing leads to wrong dimensions
}



/*
void Orchestrator::PreprocessImgGPU(const cv::Mat& img, cv::Mat& out_chw)
{
    // Upload to GPU
    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(img);

    // BGR to RGB
    cv::cuda::GpuMat rgb_img;
    cv::cuda::cvtColor(gpu_img, rgb_img, cv::COLOR_BGR2RGB);

    // Resize to 224x224
    cv::cuda::GpuMat resized_img;
    cv::cuda::resize(rgb_img, resized_img, cv::Size(224, 224), 0, 0, cv::INTER_AREA);

    // Convert to float32 and scale to [0, 1]
    cv::cuda::GpuMat float_img;
    resized_img.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // Split channels
    std::vector<cv::cuda::GpuMat> gpu_channels;
    cv::cuda::split(float_img, gpu_channels);

    // Normalize per channel
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; ++c) {
        cv::cuda::subtract(gpu_channels[c], mean[c], gpu_channels[c]);
        cv::cuda::divide(gpu_channels[c], std[c], gpu_channels[c]);
    }

    // Stack back to CHW manually
    std::vector<cv::Mat> cpu_channels(3);
    for (int c = 0; c < 3; ++c) {
        gpu_channels[c].download(cpu_channels[c]);
    }
    cv::vconcat(cpu_channels, out_chw); // (3*224)x224 → CHW
}
*/

EC Orchestrator::ExecFullInference()
{

    if (!current_frame_) 
    {
        spdlog::error("No frame to process");
        LogError(EC::NN_NO_FRAME_AVAILABLE);
        return EC::NN_NO_FRAME_AVAILABLE;
    }

    if (!rc_net_.IsInitialized()) 
    {
        spdlog::error("RCNet is not initialized. Cannot perform inference.");
        LogError(EC::NN_ENGINE_NOT_INITIALIZED);
        return EC::NN_ENGINE_NOT_INITIALIZED;
    }

    if (num_inference_performed_on_current_frame_ > 0) 
    {
        spdlog::warn("Inference already performed on the current frame. This will overwrite.");
        // TODO: clear
    }

    img_buff_ = current_frame_->GetImg();
    cv::Mat chw_img;

    // Preprocess the image
    PreprocessImg(img_buff_, chw_img);
    spdlog::info("Image preprocessed to CHW format with shape: {}x{}x{}", chw_img.rows, chw_img.cols, chw_img.channels());

    // Run the RC net 
    auto host_output = std::unique_ptr<float[]>{new float[RC_NUM_CLASSES]};
    EC rc_inference_status = rc_net_.Infer(chw_img.data, host_output.get());
    if (rc_inference_status != EC::OK) 
    {
        spdlog::error("RCNet inference failed with error code: {}", to_uint8(rc_inference_status));
        LogError(rc_inference_status);
        return rc_inference_status;
    }

    // Populate the RC ID to original frame
    static constexpr float RC_THRESHOLD = 0.5f; // Threshold for region classification
    for (uint8_t i = 0; i < RC_NUM_CLASSES; i++) 
    {
        spdlog::info("RC output for class {}: {:.3f}", i, host_output[i]);
        if (host_output[i] > RC_THRESHOLD) 
        {
            original_frame_->AddRegion(static_cast<RegionID>(i)); 
        }
    }



    // TODO: LD selection

    // TODO: LD inference 

    // TODO: Populate landmarks


    num_inference_performed_on_current_frame_++;

    return EC::OK;
}


} // namespace Inference