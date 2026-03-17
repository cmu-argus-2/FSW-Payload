#include "inference/orchestrator.hpp"
#include "spdlog/spdlog.h"

#include <filesystem>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/cuda.hpp>
// #include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace cv::dnn;

namespace Inference
{

Orchestrator::Orchestrator()
: current_frame_(nullptr),
original_frame_(nullptr),
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
    current_frame_ = nullptr;
    original_frame_ = nullptr;
}

void Orchestrator::LoadEngines()
{
    LoadRCEngine();
    LoadLDNetEngines();
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

    if (use_trt_for_ld_)
    {
        engine_path = ld_engine_folder_path_ + "/" + region_str + "/" + region_str + "_weights.trt";
    } else {
        engine_path = ld_engine_folder_path_ + "/" + region_str + "/" + region_str + "_weights.onnx";
    }

    if (!std::filesystem::exists(engine_path))
    {
        spdlog::error("Cannot initialize LDNet for region {}. Missing assets. Engine does not exist: {}", 
                    region_str, std::filesystem::exists(engine_path));
        return EC::NN_FAILED_TO_OPEN_ENGINE_FILE;
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

void Orchestrator::SetRCNetEnginePath(const std::string& path)
{
    if (!std::filesystem::exists(path))
    {
        spdlog::error("RC Net engine file not found at path: {}", path);
        return;
    }
    rc_engine_path_ = path;

    if (rc_net_.IsInitialized())
    {
        FreeRCNet();
    }

    if (preload_rc_engine_)
    {
        LoadRCEngine();
    }
}

void Orchestrator::SetLDNetFolderPath(const std::string& path)
{
    if (!std::filesystem::exists(path))
    {
        spdlog::error("LD Net engine folder not found at path: {}", path);
        return;
    }
    ld_engine_folder_path_ = path;

    InitializeLDNetRuntimes();
}

void Orchestrator::InitializeLDNetRuntimes()
{
    std::string engine_path;
    std::string csv_path;
    std::string region_str;

    EC ld_net_status;
    for(const auto& region_id : GetAllRegionIDs())
    {
        region_str = std::string(GetRegionString(region_id));

        csv_path = ld_engine_folder_path_ + "/" + region_str + "/bounding_boxes.csv";

        if (use_trt_for_ld_)
        {
            engine_path = ld_engine_folder_path_ + "/" + region_str + "/" + region_str + "_weights.trt";
        } else {
            engine_path = ld_engine_folder_path_ + "/" + region_str + "/" + region_str + "_weights.onnx";
        }

        if (!std::filesystem::exists(engine_path) || !std::filesystem::exists(csv_path))
        {
            spdlog::error("Cannot initialize LDNet for region {}. Missing assets. Engine exists: {}, CSV exists: {}", 
                        region_str, std::filesystem::exists(engine_path), std::filesystem::exists(csv_path));
            return;
        }

        if (use_trt_for_ld_)
        {
            std::unique_ptr<TRTLDNet> trt_ld_net = std::make_unique<TRTLDNet>(region_id, csv_path);
            ld_nets_[region_id] = std::move(trt_ld_net);

            spdlog::info("Loading model for region {}: Engine path: {}, CSV path: {}", region_str, engine_path, csv_path);
        }
        else // use onnx for ld
        {
            std::unique_ptr<ONNXLDNet> onnx_ld_net = std::make_unique<ONNXLDNet>(region_id, csv_path);
            ld_nets_[region_id] = std::move(onnx_ld_net);
            spdlog::info("ONNX model path set for region {}: {}", region_str, engine_path);
            // onnx will be loaded at inference time
        }
    }

    if (preload_ld_engines_)
    {
        LoadLDNetEngines(); 
    }
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

    spdlog::info("Size of out_chw_img before: {}x{}x{}", out_chw_img.rows, out_chw_img.cols, out_chw_img.channels());
    cv::vconcat(channels, out_chw_img);  // Stack channels vertically (CHW format)
    spdlog::info("Size of out_chw_img after: {}x{}x{}", out_chw_img.rows, out_chw_img.cols, out_chw_img.channels());
    // Print the first 10 pixels of the preprocessed image
    /*spdlog::info("First 10 pixels of the preprocessed image:");
    for (int i = 0; i < std::min(10, out_chw_img.rows); ++i) {
        spdlog::info("Pixel {}: {}", i, out_chw_img.at<float>(i));
    }*/
}

// Source: https://stackoverflow.com/questions/28562401/resize-an-image-to-a-square-but-keep-aspect-ratio-c-opencv
void Orchestrator::LDPreprocessImg(cv::Mat img, cv::Mat& out_chw_img, int target_width)
{
    spdlog::info("Starting LDPreprocessImg");
    cv::Mat float_img;
    img.convertTo(float_img, CV_32F, 1.0 / 255.0);  // Correctly normalize
    spdlog::info("Image converted to float32, shape: {}x{}x{}", float_img.rows, float_img.cols, float_img.channels());
    
    // TODO: letterbox the image to 4608 x 4608, or just have the LD nets be trained on the actual image size
    cv::Mat letterboxed_img = cv::Mat::zeros(target_width, target_width, float_img.type() );
    spdlog::info("Created letterboxed image with size: {}x{}x{}", letterboxed_img.rows, letterboxed_img.cols, letterboxed_img.channels());

    int max_dim = ( img.cols >= img.rows ) ? img.cols : img.rows;
    spdlog::info("Max dimension: {}", max_dim);
    
    float scale = ( ( float ) target_width ) / max_dim;
    spdlog::info("Scale factor: {}", scale);

    cv::Rect roi;
    if ( img.cols >= img.rows )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = img.rows * scale;
        roi.y = ( target_width - roi.height ) / 2;
        spdlog::info("Landscape image - ROI: x={}, y={}, width={}, height={}", roi.x, roi.y, roi.width, roi.height);
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = img.cols * scale;
        roi.x = ( target_width - roi.width ) / 2;
        spdlog::info("Portrait image - ROI: x={}, y={}, width={}, height={}", roi.x, roi.y, roi.width, roi.height);
    }
    
    cv::resize( float_img, letterboxed_img( roi ), roi.size() );
    spdlog::info("Image resized and placed in letterboxed image");
    spdlog::info("Letterboxed image shape: {}x{}x{}", letterboxed_img.rows, letterboxed_img.cols, letterboxed_img.channels());

    // int y_offset = (4608 - float_img.rows) / 2;
    // int x_offset = (4608 - float_img.cols) / 2;
    // float_img.copyTo(letterboxed_img(cv::Rect(x_offset, y_offset, float_img.cols, float_img.rows)));
    // float_img = letterboxed_img;
    
    // Convert HWC (4608x4608x3) to CHW (3x4608x4608)
    // int siz[] = {3, float_img.rows, float_img.cols};
    // out_chw_img.create(3, siz, CV_32F);
    // std::vector<cv::Mat> planes = {
    //     cv::Mat(float_img.rows, float_img.cols, float_img.type(), out_chw_img.ptr(0)), // swap 0 and 2 and you can avoid the bgr->rgb conversion !
    //     cv::Mat(float_img.rows, float_img.cols, float_img.type(), out_chw_img.ptr(1)),
    //     cv::Mat(float_img.rows, float_img.cols, float_img.type(), out_chw_img.ptr(2))
    // };
    // split(float_img, planes);
    // out_chw_img.convertTo(out_chw_img, CV_32F);
    // transposeND(letterboxed_img, {2, 0, 1}, out_chw_img); // only for single channel images

    std::vector<cv::Mat> channels(3);
    cv::split(letterboxed_img, channels);  // Split into individual float32 channels
    // std::vector<cv::Mat> chwchannels(3);
    // transposeND(hwcchannels[0], {2, 0, 1}, chwchannels[0]);
    // transposeND(hwcchannels[1], {2, 0, 1}, chwchannels[1]);
    // transposeND(hwcchannels[2], {2, 0, 1}, chwchannels[2]);
    // // spdlog::info("Split into {} channels", channels.size());
    // cv::merge(chwchannels, out_chw_img);    
    // // cv::dnn::blobFromImage(letterboxed_img, out_chw_img); // doesn't work
    cv::vconcat(channels, out_chw_img);  // Stack channels vertically (CHW format)

    spdlog::info("Stacked channels to CHW format, final shape: {}x{}x{}", out_chw_img.rows, out_chw_img.cols, out_chw_img.channels());
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

EC Orchestrator::ExecRCInference()
{
    if (!current_frame_) 
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

    if (num_inference_performed_on_current_frame_ > 0) 
    {
        spdlog::warn("Inference already performed on the current frame. This will overwrite.");
        original_frame_->ClearRegions(); // Reset per-frame RC outputs
        original_frame_->ClearLandmarks(); // Reset per-frame LD outputs
    }
    
    img_buff_ = current_frame_->GetImg();
    cv::Mat chw_img;

    // Preprocess the image
    RCPreprocessImg(img_buff_, chw_img);
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
        spdlog::info("RC output for class {}: {:.3f}", i, host_output[i]);
        if (host_output[i] > RC_THRESHOLD) 
        {
            original_frame_->AddRegion(static_cast<RegionID>(i), host_output[i]); 
        }
    }

    original_frame_->SetProcessingStage(ProcessingStage::RCNeted);

    if (!preload_rc_engine_) FreeRCNet(); // Free the RC engine if it was loaded on demand

    return EC::OK;
}

EC Orchestrator::ExecLDInferenceTRT()
{
    if (!current_frame_) 
    {
        spdlog::error("No frame to process");
        LogError(EC::NN_NO_FRAME_AVAILABLE);
        return EC::NN_NO_FRAME_AVAILABLE;
    }

    if (num_inference_performed_on_current_frame_ > 0) 
    {
        spdlog::warn("Inference already performed on the current frame. This will overwrite.");
        original_frame_->ClearLandmarks(); // Reset per-frame LD outputs
    }

    spdlog::info("Regions detected in the frame: {}", original_frame_->GetRegionIDs().size());
    std::string trt_path;
    std::string csv_path;
    // TODO: Don't copy engines, just look them up and run them directly from the map if they exist
    if (original_frame_->GetRegionIDs().empty())
    {
        spdlog::warn("No regions detected by RCNet. Skipping LD inference.");
        return EC::OK; // Not an error, just no regions to run LD on
    }

    // LD inference
    cv::Mat ld_chw_img;
    LDPreprocessImg(img_buff_, ld_chw_img);
    int output_size;

    for(const auto& region_id : original_frame_->GetRegionIDs())
    {
        if (ld_nets_.find(region_id) == ld_nets_.end())
        {
            spdlog::error("No LDNet found for region: {}", GetRegionString(region_id));
            continue;
        }

        if (!preload_ld_engines_ && !ld_nets_.at(region_id)->IsInitialized())
        {
            LoadLDNetEngineForRegion(region_id);
        }

        if (!ld_nets_.at(region_id)->IsInitialized())
        {
            spdlog::error("LDNet for region {} is not initialized.", GetRegionString(region_id));
            continue;
        }

        spdlog::info("Running LD inference for region: {}", GetRegionString(region_id));

        output_size = ld_nets_[region_id]->GetOutputSize();
        auto ld_host_output = std::unique_ptr<float[]>{new float[output_size]};
        EC ld_inference_status = ld_nets_[region_id]->Infer(ld_chw_img.data, ld_host_output.get());
        if (ld_inference_status != EC::OK) 
        {
            spdlog::error("LDNet inference failed with error code: {}", to_uint8(ld_inference_status));
            LogError(ld_inference_status);
            return ld_inference_status;
        } else {
            spdlog::info("LDNet inference completed successfully. Output size: {}", output_size);
        }

        // Non-max suppression and populate landmarks
        ld_nets_[region_id]->PostprocessOutput(ld_host_output.get(), original_frame_); // This will populate the landmarks in the original frame based on the LD output
        spdlog::info("LDNet inference output post-processed");

        if (!preload_ld_engines_) FreeLDNetForRegion(region_id); // Free the LD engine for this region if it was loaded on demand
    }
    original_frame_->SetProcessingStage(ProcessingStage::LDNeted);
    
    return EC::OK;
}

EC Orchestrator::ExecLDInferenceONNX()
{
    if (!current_frame_) 
    {
        spdlog::error("No frame to process");
        LogError(EC::NN_NO_FRAME_AVAILABLE);
        return EC::NN_NO_FRAME_AVAILABLE;
    }

    if (num_inference_performed_on_current_frame_ > 0) 
    {
        spdlog::warn("Inference already performed on the current frame. This will overwrite.");
        original_frame_->ClearLandmarks(); // Reset per-frame LD outputs
    }

    if (original_frame_->GetRegionIDs().empty())
    {
        spdlog::warn("No regions detected by RCNet. Skipping LD inference.");
        return EC::OK;
    }
    spdlog::info("Regions detected in the frame: {}", original_frame_->GetRegionIDs().size());

    // If LDNet engines are not initialized, initialize them. If they don't exist, skip them.
    std::vector<RegionID> region_ids;

    // Pre-process the image for LD
    int output_size;

    float paddingValue = 0.0f;
    bool swapRB = true; // true = rgb
    int inpWidth = 4608;
    int inpHeight = 4608;
    Scalar scale = Scalar(1.0/255.0,1.0/255.0,1.0/255.0);
    Scalar mean = Scalar(0.0,0.0,0.0);
    ImagePaddingMode paddingMode = static_cast<ImagePaddingMode>(2);

    Size size(inpWidth, inpHeight);
    Image2BlobParams imgParams(
        scale,
        size,
        mean,
        swapRB,
        CV_32F,
        DNN_LAYOUT_NCHW,
        paddingMode); // , paddingValue);

    Image2BlobParams paramNet;
    paramNet.scalefactor = scale;
    paramNet.size = size;
    paramNet.mean = mean;
    paramNet.swapRB = swapRB;
    paramNet.paddingmode = paddingMode;
    cv::dnn::Net net;
    int backend = 0; // automatically, opencv implementation or cuda?
    int target = 0; // 0: cpu, 6: cuda, 7: cuda fp16

    std::vector<Mat> outs;
    std::vector<int> keep_classIds;
    std::vector<float> keep_confidences;
    std::vector<Rect2d> keep_boxes;
    std::vector<Rect> boxes;
    spdlog::info("Forward buffers initialized");

    Mat inp;
    //![preprocess_call_func]
    inp = blobFromImageWithParams(img_buff_, imgParams);
    spdlog::info("Blob created from image");
    EC ld_net_status;
    for(const auto& region_id : original_frame_->GetRegionIDs())
    {

        if (ld_nets_.find(region_id) == ld_nets_.end())
        {
            spdlog::error("No LDNet found for region: {}", GetRegionString(region_id));
            continue;
        }

        spdlog::info("Selected LDNet for region: {}", GetRegionString(region_id));

        if (!preload_ld_engines_ && !ld_nets_.at(region_id)->IsInitialized())
        {
            LoadLDNetEngineForRegion(region_id);
        }

        if (!ld_nets_.at(region_id)->IsInitialized())
        {
            spdlog::warn("LDNet for region {} is not initialized. Attempting to load ONNX model.", GetRegionString(region_id));

            ld_net_status = LoadLDNetEngineForRegion(region_id); // Attempt to load the engine for this region
            if (ld_net_status != EC::OK) 
            {
                spdlog::error("Failed to load ONNX LD Net engine for region: {}. Skipping this region.", GetRegionString(region_id));
                continue;
            } else {
                spdlog::info("Successfully loaded ONNX LD Net engine for region: {}", GetRegionString(region_id));
            }
        }
        // Run        
        ld_net_status = ld_nets_.at(region_id)->Infer(inp, outs);

        if (ld_net_status != EC::OK) 
        {
            spdlog::error("LDNet inference failed with error code: {} for region: {}", to_uint8(ld_net_status), GetRegionString(region_id));
            LogError(ld_net_status);
            continue;
        } else {
            spdlog::info("LDNet inference completed successfully for region: {}. Output size: {}", GetRegionString(region_id), outs.size());
        }

        ld_net_status = ld_nets_.at(region_id)->PostprocessOutput(outs, original_frame_);

        // Free the LD engine for this region if it was loaded on demand
        if (!preload_ld_engines_) FreeLDNetForRegion(region_id);

    }

    original_frame_->SetProcessingStage(ProcessingStage::LDNeted);


    return EC::OK;
}


EC Orchestrator::ExecFullInference()
{
    // Run Region Classification inference
    EC rc_ec = ExecRCInference();

    if (rc_ec != EC::OK)  return rc_ec;
    
    // Run Landmark Detection inference
    // TODO: Functions below should be merged
    if (use_trt_for_ld_)
    {
        EC ld_ec = ExecLDInferenceTRT();
        num_inference_performed_on_current_frame_++;
        return ld_ec;
    }

    EC ld_ec = ExecLDInferenceONNX();
    num_inference_performed_on_current_frame_++;

    return ld_ec;
}


} // namespace Inference