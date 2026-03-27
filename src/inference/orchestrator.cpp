#include "inference/orchestrator.hpp"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace cv::dnn;

namespace Inference
{

Orchestrator::Orchestrator()
: original_frame_(nullptr),
rc_net_()
{
    SetRCNetEnginePath(rc_engine_path_);
    SetLDNetEngineFolderPath(ld_engine_folder_path_);
}

void Orchestrator::FreeEngines()
{
    FreeRCNet();
    FreeLDNets();
    spdlog::info("Orchestrator engines freed.");
}

void Orchestrator::FreeRCNet()
{
    rc_net_.Free();
}

void Orchestrator::FreeLDNets()
{
    for (auto& [region_id, ld_net] : ld_nets_)
    {
        if (ld_net)
        {
            ld_net->Free();
        }
    }
}

void Orchestrator::FreeLDNetForRegion(RegionID region_id) 
{
    if (ld_nets_.find(region_id) != ld_nets_.end() && ld_nets_[region_id]) 
    {
        ld_nets_[region_id]->Free();
        spdlog::info("Freed LDNet for region: {}", GetRegionString(region_id));
    }
}


Orchestrator::~Orchestrator()
{
    FreeEngines();
    original_frame_ = nullptr;
}

EC Orchestrator::LoadRCEngine()
{
    EC rc_net_status = rc_net_.LoadEngine(rc_engine_path_);
    if (rc_net_status != EC::OK) 
    {
        spdlog::error("Failed to load RC Net engine.");
    }
    return rc_net_status;
}

void Orchestrator::LoadLDNetEngines()
{
    EC ld_net_status;
    size_t loaded_ld_nets = 0;
    for(const auto& region_id : GetAllRegionIDs())
    {
        ld_net_status = LoadLDNetEngineForRegion(region_id);
        if (ld_net_status == EC::OK) loaded_ld_nets++;
    }

    if (loaded_ld_nets == 0)
    {
        spdlog::warn("No LD models were loaded from folder: {}", ld_engine_folder_path_);
    }
    else
    {
        spdlog::info("Loaded {} LD model(s) from folder: {}", loaded_ld_nets, ld_engine_folder_path_);
    }

}

EC Orchestrator::LoadLDNetEngineForRegion(RegionID region_id) 
{
    std::string region_str = std::string(GetRegionString(region_id));
    std::string engine_path;

    engine_path = ld_engine_folder_path_ + "/" + region_str + "/" + region_str + ldnet_config.GetFileNameAppendix();


    if (!std::filesystem::exists(engine_path))
    {
        spdlog::error("Cannot initialize LDNet for region {}. Missing assets. Engine does not exist: {}",
                    region_str, engine_path);
        return EC::NN_FAILED_TO_OPEN_ENGINE_FILE;
    }

    if (ld_nets_.find(region_id) == ld_nets_.end())
    {
        spdlog::error("No LDNet runtime for region {}. Call InitializeLDNetRuntimes first.", region_str);
        return EC::NN_ENGINE_NOT_INITIALIZED;
    }

    EC ld_net_status = ld_nets_.at(region_id)->LoadEngine(engine_path);

    if (ld_net_status != EC::OK) 
    {
        spdlog::error("Failed to load LD Net engine for region: {}", region_str);
    } else {
        spdlog::info("LDNet successfully loaded for region: {}", region_str);
    }
    return ld_net_status;
}


void Orchestrator::GrabNewImage(std::shared_ptr<Frame> frame)
{
    if (!frame)
    {
        spdlog::error("Received null frame.");
        return;
    }

    original_frame_ = frame;
    num_rc_inferences_on_current_frame_ = 0;
    num_ld_inferences_on_current_frame_ = 0;
}

size_t Orchestrator::GetMemorySize(const nvinfer1::Dims& dims, size_t element_size)
{
    size_t size = element_size;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0)
        {
            spdlog::error("GetMemorySize: dynamic dimension at index {} (value {}). Resolve shape before calling.", i, dims.d[i]);
            return 0;
        }
        size *= static_cast<size_t>(dims.d[i]);
    }
    return size;
}

EC Orchestrator::SetRCNetEnginePath(const std::string& path)
{
    if (!std::filesystem::exists(path))
    {
        spdlog::error("RC Net engine file not found at path: {}", path);
        return EC::FILE_DOES_NOT_EXIST;
    }
    rc_engine_path_ = path;

    if (rc_net_.IsInitialized())
    {
        FreeRCNet();
    }

    if (preload_rc_engine_)
    {
        return LoadRCEngine();
    }
    return EC::OK;
}

EC Orchestrator::SetLDNetEngineFolderPath(const std::string& path)
{
    if (!std::filesystem::exists(path))
    {
        spdlog::error("LD Net engine folder not found at path: {}", path);
        return EC::FILE_DOES_NOT_EXIST;
    }
    ld_engine_folder_path_ = path;

    InitializeLDNetRuntimes();
    return EC::OK;
}

void Orchestrator::InitializeLDNetRuntimes()
{
    FreeLDNets();
    ld_nets_.clear();

    std::string engine_path;
    std::string csv_path;
    std::string region_str;

    EC ld_net_status;
    for(const auto& region_id : GetAllRegionIDs())
    {
        region_str = std::string(GetRegionString(region_id));

        csv_path = ld_engine_folder_path_ + "/" + region_str + "/bounding_boxes.csv";

        engine_path = ld_engine_folder_path_ + "/" + region_str + "/" + region_str + ldnet_config.GetFileNameAppendix();

        if (!std::filesystem::exists(engine_path) || !std::filesystem::exists(csv_path))
        {
            spdlog::error("Cannot initialize LDNet for region {}. Missing assets. Engine exists: {}, CSV exists: {}", 
                        region_str, std::filesystem::exists(engine_path), std::filesystem::exists(csv_path));
            continue;
        }

        // TODO: merge both below
        std::unique_ptr<LDNet> ld_net = std::make_unique<LDNet>(region_id, csv_path);
        ld_nets_[region_id] = std::move(ld_net);
        spdlog::info("Loading model for region {}: Engine path: {}, CSV path: {}", region_str, engine_path, csv_path);
    }

    if (preload_ld_engines_)
    {
        LoadLDNetEngines(); 
    }
}

void Orchestrator::SetLDNetConfig(NET_QUANTIZATION weight_quant, int input_width, int input_height, bool embedded_nms, bool use_trt_for_ld)
{
    ldnet_config = {weight_quant, input_width, input_height, embedded_nms, use_trt_for_ld};
    InitializeLDNetRuntimes(); // config change invalidates the existing ld_nets_ map
}

void Orchestrator::RCPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img)
{
   if (img.empty())
    {
        spdlog::error("RCPreprocessImg: input image is empty.");
        out_chw_img.release();
        return;
    }

    // Convert BGR -> RGB explicitly
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

    const cv::Scalar scalefactor(
        1.0 / (255.0 * 0.229),  // R
        1.0 / (255.0 * 0.224),  // G
        1.0 / (255.0 * 0.225),  // B
        0.0
    );

    const cv::Scalar mean(
        255.0 * 0.485,          // R
        255.0 * 0.456,          // G
        255.0 * 0.406,          // B
        0.0
    );

    cv::dnn::Image2BlobParams params(
        scalefactor,
        Size(224, 224),
        mean,
        false,                  // swapRB = false because we already converted to RGB
        CV_32F,
        DNN_LAYOUT_NCHW
    );

    blobFromImageWithParams(rgb_img, out_chw_img, params);

}

/*
void Orchestrator::RCPreprocessImgGPU(const cv::Mat& img, cv::Mat& out_chw)
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

// Source: https://stackoverflow.com/questions/28562401/resize-an-image-to-a-square-but-keep-aspect-ratio-c-opencv
void Orchestrator::LDPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img, int target_width, int target_height)
{
    bool swapRB = true; // true = rgb
    Scalar scale = Scalar(1.0/255.0,1.0/255.0,1.0/255.0);
    Scalar mean = Scalar(0.0,0.0,0.0);
    ImagePaddingMode paddingMode = DNN_PMODE_LETTERBOX;

    Size size(target_width, target_height);
    Image2BlobParams imgParams(
        scale,
        size,
        mean,
        swapRB,
        CV_32F,
        DNN_LAYOUT_NCHW, // DNN_LAYOUT_NHWC
        paddingMode);

    out_chw_img = blobFromImageWithParams(img, imgParams);
}


EC Orchestrator::ExecRCInference()
{
    if (!original_frame_)
    {
        spdlog::error("No frame to process");
        LogError(EC::NN_NO_FRAME_AVAILABLE);
        return EC::NN_NO_FRAME_AVAILABLE;
    }

    if (!preload_rc_engine_ && !rc_net_.IsInitialized()) 
    {
        LoadRCEngine();
    }

    if (!rc_net_.IsInitialized()) 
    {
        spdlog::error("RCNet is not initialized. Cannot perform inference.");
        LogError(EC::NN_ENGINE_NOT_INITIALIZED);
        return EC::NN_ENGINE_NOT_INITIALIZED;
    }

    if (num_rc_inferences_on_current_frame_ > 0)
    {
        spdlog::warn("RC inference already performed on the current frame. Overwriting.");
        original_frame_->ClearRegions();
        original_frame_->ClearLandmarks();
        num_ld_inferences_on_current_frame_ = 0; // LD results are now invalid
    }

    cv::Mat img = original_frame_->GetImg();
    cv::Mat chw_img;

    // Preprocess the image
    RCPreprocessImg(img, chw_img);
    spdlog::info("Image preprocessed to CHW format with shape: {}x{}x{}", chw_img.rows, chw_img.cols, chw_img.channels());

    // Run the RC net 
    int rc_num_classes = rc_net_.GetNumClasses();
    auto host_output = std::unique_ptr<float[]>{new float[rc_num_classes]};
    EC rc_inference_status = rc_net_.Infer(chw_img.data, host_output.get());
    if (rc_inference_status != EC::OK) 
    {
        spdlog::error("RCNet inference failed with error code: {}", to_uint8(rc_inference_status));
        LogError(rc_inference_status);
        if (!preload_rc_engine_) FreeRCNet(); // Free the RC engine if it was loaded on demand
        return rc_inference_status;
    }

    // Populate the RC ID to original frame
    static constexpr float RC_THRESHOLD = 0.5f; // Threshold for region classification TODO Change this later on validation
    for (uint8_t i = 0; i < rc_num_classes; i++) 
    {
        if (host_output[i] > RC_THRESHOLD)
        {
            spdlog::info("RC: class {} score {:.3f} above threshold", GetRegionString(static_cast<RegionID>(i)), host_output[i]);
            original_frame_->AddRegion(static_cast<RegionID>(i), host_output[i]);
        }
    }

    original_frame_->SetProcessingStage(ProcessingStage::RCNeted);
    num_rc_inferences_on_current_frame_++;

    if (!preload_rc_engine_) FreeRCNet(); // Free the RC engine if it was loaded on demand

    return EC::OK;
}

EC Orchestrator::ExecLDInference()
{
    if (!original_frame_)
    {
        spdlog::error("No frame to process");
        LogError(EC::NN_NO_FRAME_AVAILABLE);
        return EC::NN_NO_FRAME_AVAILABLE;
    }

    if (num_ld_inferences_on_current_frame_ > 0)
    {
        spdlog::warn("LD inference already performed on the current frame. Overwriting.");
        original_frame_->ClearLandmarks();
    }

    if (original_frame_->GetRegionIDs().empty())
    {
        spdlog::warn("No regions detected by RCNet. Skipping LD inference.");
        return EC::OK; // Not an error, just no regions to run LD on
    }

    spdlog::info("Regions detected in the frame: {}", original_frame_->GetRegionIDs().size());

    cv::Mat img = original_frame_->GetImg();

    // LD inference
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat ld_chw_img;
    LDPreprocessImg(img, ld_chw_img, ldnet_config.input_width, ldnet_config.input_height);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::info("LD preprocessing (took {} ms)", duration.count());
    
    int output_size;
    EC ld_net_status;
    for(const auto& region_id : original_frame_->GetRegionIDs())
    {
        if (ld_nets_.find(region_id) == ld_nets_.end())
        {
            spdlog::error("No LDNet found for region: {}", GetRegionString(region_id));
            continue;
        }

        if (!preload_ld_engines_ && !ld_nets_.at(region_id)->IsInitialized())
        {
            start = std::chrono::high_resolution_clock::now();
            ld_net_status = LoadLDNetEngineForRegion(region_id);
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            spdlog::info("LD preloading (took {} ms)", duration.count());
        }

        if (!ld_nets_.at(region_id)->IsInitialized())
        {
            spdlog::warn("LDNet for region {} is not initialized. Attempting to load model.", GetRegionString(region_id));

            ld_net_status = LoadLDNetEngineForRegion(region_id); // Attempt to load the engine for this region
            if (ld_net_status != EC::OK) 
            {
                spdlog::error("Failed to load LD Net engine for region: {}. Skipping this region.", GetRegionString(region_id));
                continue;
            } else {
                spdlog::info("Successfully loaded LD Net engine for region: {}", GetRegionString(region_id));
            }
        }

        spdlog::info("Running LD inference for region: {}", GetRegionString(region_id));

        start = std::chrono::high_resolution_clock::now();
        
        output_size = ld_nets_[region_id]->GetOutputSize();
        if (ld_nets_[region_id]->IsTRT()) {
            float* raw_out = ld_nets_[region_id]->GetOutputBuffer();
            ld_net_status = ld_nets_[region_id]->Infer(ld_chw_img.data, raw_out);

            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            spdlog::info("Inference (took {} ms)", duration.count());
            
            if (ld_net_status != EC::OK) 
            {
                spdlog::error("LDNet inference failed with error code: {}", to_uint8(ld_net_status));
                LogError(ld_net_status);
                ld_nets_[region_id]->ReleaseScratchBuffers();
                return ld_net_status;
            } else {
                spdlog::info("LDNet inference completed successfully. Output size: {}", output_size);
            }

            // Raw output buffer may be lazily allocated during Infer().
            raw_out = ld_nets_[region_id]->GetOutputBuffer();
            if (!raw_out)
            {
                spdlog::error("LDNet output buffer is null after inference.");
                LogError(EC::NN_POINTER_NULL);
                ld_nets_[region_id]->ReleaseScratchBuffers();
                return EC::NN_POINTER_NULL;
            }

            int sizes[3] = {1, ld_nets_[region_id]->GetNumLandmarks() + 4, ld_nets_[region_id]->GetNumYoloBoxes()};
            cv::Mat output_matrix(3, sizes, CV_32F, raw_out);

            start = std::chrono::high_resolution_clock::now();
            EC postprocess_status = ld_nets_[region_id]->PostprocessOutput(output_matrix, original_frame_); // This will populate the landmarks in the original frame based on the LD output
            ld_nets_[region_id]->ReleaseScratchBuffers();

            if (postprocess_status != EC::OK)
            {
                spdlog::error("LDNet postprocess failed with error code: {}", to_uint8(postprocess_status));
                LogError(postprocess_status);
                return postprocess_status;
            }

        } else {
            std::vector<cv::Mat> outs;
            ld_net_status = ld_nets_.at(region_id)->Infer(ld_chw_img, outs);
            
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            spdlog::info("Inference (took {} ms)", duration.count());
            
            if (ld_net_status != EC::OK) 
            {
                spdlog::error("LDNet inference failed with error code: {}", to_uint8(ld_net_status));
                LogError(ld_net_status);
                return ld_net_status;
            } else {
                spdlog::info("LDNet inference completed successfully. Output size: {}", output_size);
            }
            start = std::chrono::high_resolution_clock::now();
            EC postprocess_status = ld_nets_[region_id]->PostprocessOutput(outs[0], original_frame_); // This will populate the landmarks in the original frame based on the LD output

            if (postprocess_status != EC::OK)
            {
                spdlog::error("LDNet postprocess failed with error code: {}", to_uint8(postprocess_status));
                LogError(postprocess_status);
                return postprocess_status;
            }

        }
        spdlog::info("LDNet inference output post-processed");
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        spdlog::info("Post-processing (took {} ms)", duration.count());

        if (!preload_ld_engines_) FreeLDNetForRegion(region_id); // Free the LD engine for this region if it was loaded on demand
    }
    original_frame_->SetProcessingStage(ProcessingStage::LDNeted);
    num_ld_inferences_on_current_frame_++;

    return EC::OK;
}


EC Orchestrator::ExecFullInference()
{
    EC rc_ec = ExecRCInference();
    if (rc_ec != EC::OK) return rc_ec;

    return ExecLDInference();
}


} // namespace Inference