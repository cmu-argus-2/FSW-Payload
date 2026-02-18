#include "inference/orchestrator.hpp"
#include "spdlog/spdlog.h"

#include <filesystem>
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

void Orchestrator::Initialize(const std::string& rc_engine_path, const std::string& ld_engine_folder_path)
{
    // Initialize the RCNet runtime
    EC rc_net_status = rc_net_.LoadEngine(rc_engine_path);
    if (rc_net_status != EC::OK) 
    {
        spdlog::error("Failed to load RC Net engine.");
        return; // could have fallback in which we still run the LDs but that is out of scope for now
    }

    // Find all the regions
    std::string trt_path;
    std::string csv_path;

    EC ld_net_status;
    size_t loaded_ld_nets = 0;
    for(const auto& region_id : GetAllRegionIDs())
    {
        std::string region_str = std::string(GetRegionString(region_id));

        trt_path = ld_engine_folder_path + "/" + region_str + "/" + region_str + "_weights.trt";
        csv_path = ld_engine_folder_path + "/" + region_str + "/bounding_boxes.csv";

        if (!std::filesystem::exists(trt_path) || !std::filesystem::exists(csv_path))
        {
            spdlog::warn("Skipping region {} (missing LD assets). TRT exists: {}, CSV exists: {}", 
                        region_str, std::filesystem::exists(trt_path), std::filesystem::exists(csv_path));
            continue;
        }
        
        ld_nets_.try_emplace(region_id, region_id, csv_path);

        spdlog::info("Loading model for region {}: TRT path: {}, CSV path: {}", region_str, trt_path, csv_path);
        ld_net_status = ld_nets_.at(region_id).LoadEngine(trt_path);
        if (ld_net_status != EC::OK) 
        {
            spdlog::error("Failed to load LD Net engine for region: {}", region_str);
            continue;
        }

        spdlog::info("LDNet successfully loaded for region: {}", region_str);
        loaded_ld_nets++;
    }

    if (loaded_ld_nets == 0)
    {
        spdlog::warn("No LD models were loaded from folder: {}", ld_engine_folder_path);
    }
    else
    {
        spdlog::info("Loaded {} LD model(s) from folder: {}", loaded_ld_nets, ld_engine_folder_path);
    }

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

void Orchestrator::RCPreprocessImg(cv::Mat img, cv::Mat& out_chw_img)
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
    // Print the first 10 pixels of the preprocessed image
    /*spdlog::info("First 10 pixels of the preprocessed image:");
    for (int i = 0; i < std::min(10, out_chw_img.rows); ++i) {
        spdlog::info("Pixel {}: {}", i, out_chw_img.at<float>(i));
    }*/
}

void Orchestrator::LDPreprocessImg(cv::Mat img, cv::Mat& out_chw_img)
{
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0 / 255.0);  // Correctly normalize
    // TODO: letterbox the image to 4608 x 4608, or just have the LD nets be trained on the actual image size
    
    // Convert HWC (224x224x3) to CHW (3x224x224)
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);  // Split into individual float32 channels
    cv::vconcat(channels, out_chw_img);  // Stack channels vertically (CHW format)
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
    original_frame_->ClearRegions(); // Reset per-frame RC outputs
    original_frame_->ClearLandmarks(); // Reset per-frame LD outputs

    img_buff_ = current_frame_->GetImg();
    cv::Mat chw_img;

    // Preprocess the image
    RCPreprocessImg(img_buff_, chw_img);
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
    static constexpr float RC_THRESHOLD = 0.5f; // Threshold for region classification TODO Change this later on validation
    for (uint8_t i = 0; i < RC_NUM_CLASSES; i++) 
    {
        spdlog::info("RC output for class {}: {:.3f}", i, host_output[i]);
        if (host_output[i] > RC_THRESHOLD) 
        {
            original_frame_->AddRegion(static_cast<RegionID>(i), host_output[i]); 
        }
    }

    original_frame_->SetProcessingStage(ProcessingStage::RCNeted);

    // TODO: Recheck LD selection process
    spdlog::info("Regions detected in the frame: {}", original_frame_->GetRegionIDs().size());
    std::string trt_path;
    std::string csv_path;
    LDNet* ld_net = nullptr;
    if (original_frame_->GetRegionIDs().empty())
    {
        spdlog::warn("No regions detected by RCNet. Skipping LD inference.");
        return EC::OK; // Not an error, just no regions to run LD on
    }
    for(const auto& region_id : original_frame_->GetRegionIDs())
    {
        if (ld_nets_.find(region_id) == ld_nets_.end())
        {
            spdlog::error("No LDNet found for region: {}", GetRegionString(region_id));
            continue;
        }
        if (!ld_nets_.at(region_id).IsInitialized())
        {
            spdlog::error("LDNet for region {} is not initialized.", GetRegionString(region_id));
            continue;
        }
        ld_net = &ld_nets_.at(region_id);
        
        spdlog::info("Selected LDNet for region: {}", GetRegionString(region_id));
        break;
    }

    if (!ld_net)
    {
        spdlog::warn("No initialized LDNet matches RC output regions. Skipping LD inference for this frame.");
        return EC::OK;
    }

    // LD inference
    cv::Mat ld_chw_img;
    LDPreprocessImg(img_buff_, ld_chw_img);

    int output_size = ld_net->GetOutputSize();
    auto ld_host_output = std::unique_ptr<float[]>{new float[output_size]};
    EC ld_inference_status = ld_net->Infer(ld_chw_img.data, ld_host_output.get());
    if (ld_inference_status != EC::OK) 
    {
        spdlog::error("LDNet inference failed with error code: {}", to_uint8(ld_inference_status));
        LogError(ld_inference_status);
        return ld_inference_status;
    } else {
        spdlog::info("LDNet inference completed successfully. Output size: {}", output_size);
    }

    // Non-max suppression and populate landmarks
    ld_net->PostprocessOutput(ld_host_output.get(), original_frame_); // This will populate the landmarks in the original frame based on the LD output
    original_frame_->SetProcessingStage(ProcessingStage::LDNeted);

    num_inference_performed_on_current_frame_++;

    return EC::OK;
}


} // namespace Inference