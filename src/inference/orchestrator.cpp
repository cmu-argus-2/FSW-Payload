#include "inference/orchestrator.hpp"
#include "spdlog/spdlog.h"

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
// #include <opencv2/core/cuda.hpp>
// #include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudaimgproc.hpp>

namespace Inference
{

Orchestrator::Orchestrator()
// : current_frame_(nullptr)
{

}

void Orchestrator::FreeEngines()
{
    rc_net_.Free();
    ld_nets_.clear();
    
    //Drop persistent preprocessing scratch memory when orchestrator is freed
    ld_letterboxed_u8_.release();
    ld_chw_buffer_.release();
    img_buff_.release();

    // current_frame_ = nullptr;
    original_frame_ = nullptr;
    spdlog::info("Orchestrator engines freed.");
}

Orchestrator::~Orchestrator()
{
    FreeEngines();
}

void Orchestrator::Initialize(const std::string& rc_engine_path, const std::string& ld_engine_folder_path)
{
    // Initialize the RCNet runtime
    EC rc_net_status = rc_net_.LoadEngine(rc_engine_path);
    if (rc_net_status != EC::OK)
    {
        spdlog::error("Failed to load RC Net engine.");
        return;
    }

    // Scan the folder and register asset paths, but don't load the LD yet
    // NOTE: Engines are loaded lazily in ExecLDInference() when RC first selects a region. Does this actually help?
    size_t registered = 0;
    for (const auto& region_id : GetAllRegionIDs())
    {
        const std::string region_str = std::string(GetRegionString(region_id));
        const std::string trt_path = ld_engine_folder_path + "/" + region_str + "/" + region_str + "_weights.trt";
        const std::string csv_path = ld_engine_folder_path + "/" + region_str + "/bounding_boxes.csv";

        if (!std::filesystem::exists(trt_path) || !std::filesystem::exists(csv_path))
        {
            spdlog::warn("Skipping region {} (missing LD assets). TRT exists: {}, CSV exists: {}",
                         region_str, std::filesystem::exists(trt_path), std::filesystem::exists(csv_path));
            continue;
        }

        ld_assets_[region_id] = {trt_path, csv_path};
        spdlog::info("Registered LD assets for region {} (engine will be loaded on first use).", region_str);
        registered++;
    }

    if (registered == 0)
        spdlog::warn("No LD assets found in folder: {}", ld_engine_folder_path);
    else
        spdlog::info("Registered {} LD asset(s). No GPU memory allocated yet.", registered);
}

void Orchestrator::GrabNewImage(std::shared_ptr<Frame> frame)
{
    if (!frame) 
    {
        spdlog::error("Received null frame.");
        return;
    }

    original_frame_ = frame;
    num_inference_performed_on_current_frame_ = 0;
}

size_t Orchestrator::GetMemorySize(const nvinfer1::Dims& dims, size_t element_size)
{
    size_t size = element_size;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}

void Orchestrator::RCPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img)
{
    // TODO: optimize and remove copies. Leverage GPU

    // Convert BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

    // 1. Resize
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(RC_INPUT_WIDTH, RC_INPUT_HEIGHT), 0, 0, cv::INTER_AREA);
    // Note: cv::INTER_AREA is the best one I found closest to torchvision.transforms.Resize (which uses PIL resize's default)
    // Ideally, we would want to avoid that because it has *slight* pixel differences due to their implementation
    // spdlog::info("Image resized to 224x224, current shape: {}x{}x{}", img.rows, img.cols, img.channels());

    // 2. ToTensor (Convert to float and scale to [0, 1]) 
    // Convert from uint8 [0, 255] to float32 [0, 1]
    cv::Mat float_img;
    resized_img.convertTo(float_img, CV_32F, 1.0 / 255.0);  // Correctly normalize
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
    cv::vconcat(channels, out_chw_img);
}

// Source: https://stackoverflow.com/questions/28562401/resize-an-image-to-a-square-but-keep-aspect-ratio-c-opencv
void Orchestrator::LDPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img, int target_width)
{
    spdlog::info("Starting LDPreprocessImg");
    
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
    // TODO: letterbox the image to 4608 x 4608, or just have the LD nets be trained on the actual image size
    // Reusing the letterbox from earlier saves some memory
    if (ld_letterboxed_u8_.size() != cv::Size(target_width, target_width) || ld_letterboxed_u8_.type() != img.type())
    {
        ld_letterboxed_u8_ = cv::Mat::zeros(target_width, target_width, img.type());
        spdlog::info("Initialized letterbox buffer with size: {}x{} and type: {}", ld_letterboxed_u8_.cols, ld_letterboxed_u8_.rows, ld_letterboxed_u8_.type());
    }

    cv::resize(img, ld_letterboxed_u8_(roi), roi.size());
    // cv::Mat letterboxed_img = cv::Mat::zeros(target_width, target_width, float_img.type() );
    // spdlog::info("Created letterboxed image with size: {}x{}x{}", letterboxed_img.rows, letterboxed_img.cols, letterboxed_img.channels());
    

    // We can reuse the ld_chw_buffer instead of float_img
    // cv::Mat float_img;
    // img.convertTo(float_img, CV_32F, 1.0 / 255.0);  // Correctly normalize
    // spdlog::info("Image converted to float32, shape: {}x{}x{}", float_img.rows, float_img.cols, float_img.channels());

    // cv::resize( float_img, letterboxed_img( roi ), roi.size() );

    if (ld_chw_buffer_.rows != 3 * target_width || ld_chw_buffer_.cols != target_width)
        ld_chw_buffer_.create(3 * target_width, target_width, CV_32F);

    spdlog::info("Image resized and placed in letterboxed image");
    // spdlog::info("Letterboxed image shape: {}x{}x{}", letterboxed_img.rows, letterboxed_img.cols, letterboxed_img.channels());
    spdlog::info("Letterboxed image shape: {}x{}x{}", ld_letterboxed_u8_.rows, ld_letterboxed_u8_.cols, ld_letterboxed_u8_.channels());

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

    // This approach creates several big buffers
    // std::vector<cv::Mat> channels(3);
    // cv::split(letterboxed_img, channels);  // Split into individual float32 channels
    // std::vector<cv::Mat> chwchannels(3);
    // transposeND(hwcchannels[0], {2, 0, 1}, chwchannels[0]);
    // transposeND(hwcchannels[1], {2, 0, 1}, chwchannels[1]);
    // transposeND(hwcchannels[2], {2, 0, 1}, chwchannels[2]);
    // // spdlog::info("Split into {} channels", channels.size());
    // cv::merge(chwchannels, out_chw_img);    
    // // cv::dnn::blobFromImage(letterboxed_img, out_chw_img); // doesn't work

    // NOTE: This could probably be done in a way that doesn't create a buffer copy 
    // cv::vconcat(channels, out_chw_img);  // Stack channels vertically (CHW format) 

    // This does not create intermediate buffers
    // Creates temp buffer u8_chan, iterates through channels, and converts directly into slice of output CHW buffer
    cv::Mat u8_chan;
    for (int c = 0; c < 3; ++c)
    {
        cv::extractChannel(ld_letterboxed_u8_, u8_chan, c);
        cv::Mat chan_roi = ld_chw_buffer_.rowRange(c * target_width, (c + 1) * target_width);
        u8_chan.convertTo(chan_roi, CV_32F, 1.0 / 255.0);
    }

    // Note this acts more like a pointer assignment than a copy per how cv::Mat works, this avoids big data buffer 
    out_chw_img = ld_chw_buffer_;

    spdlog::info("Stacked channels to CHW format, final shape: {}x{}x{}", out_chw_img.rows, out_chw_img.cols, out_chw_img.channels());
}


// void Orchestrator::LDPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img, int target_width)
// {
//     // Memory layout goal: avoid the ~964 MB peak of the naive approach which kept
//     // float_img (241 MB) + letterboxed_img (241 MB) + split channels (241 MB) +
//     // vconcat output (241 MB) all live simultaneously.
//     //
//     // Instead:
//     //  1. Letterbox the uint8 source (60 MB) into a persistent buffer — no float copy.
//     //  2. Extract each uint8 channel and convertTo float directly into the
//     //     persistent CHW buffer (241 MB).  Peak: ~320 MB total.

//     const int max_dim = std::max(img.cols, img.rows);
//     const float scale = (float)target_width / max_dim;

//     cv::Rect roi;
//     if (img.cols >= img.rows)
//     {
//         roi.width  = target_width;
//         roi.x      = 0;
//         roi.height = (int)(img.rows * scale);
//         roi.y      = (target_width - roi.height) / 2;
//     }
//     else
//     {
//         roi.y      = 0;
//         roi.height = target_width;
//         roi.width  = (int)(img.cols * scale);
//         roi.x      = (target_width - roi.width) / 2;
//     }

//     // Reuse the persistent uint8 letterbox canvas (~60 MB). Zero-fill on first use
//     // or size change; reuse directly otherwise (cv::resize writes the ROI, black
//     // padding outside the ROI persists from initialization).
//     if (ld_letterboxed_u8_.size() != cv::Size(target_width, target_width)
//         || ld_letterboxed_u8_.type() != img.type())
//     {
//         ld_letterboxed_u8_ = cv::Mat::zeros(target_width, target_width, img.type());
//     }
//     cv::resize(img, ld_letterboxed_u8_(roi), roi.size());

//     // Reuse the persistent float32 CHW output buffer (~241 MB).
//     // Layout: rows [0,H) = channel 0, [H,2H) = channel 1, [2H,3H) = channel 2.
//     if (ld_chw_buffer_.rows != 3 * target_width || ld_chw_buffer_.cols != target_width)
//         ld_chw_buffer_.create(3 * target_width, target_width, CV_32F);

//     // Convert each uint8 channel directly into its slice of the CHW buffer.
//     // u8_chan (~20 MB) is the only temporary allocation; freed each iteration.
//     cv::Mat u8_chan;
//     for (int c = 0; c < 3; ++c)
//     {
//         cv::extractChannel(ld_letterboxed_u8_, u8_chan, c);
//         cv::Mat chan_roi = ld_chw_buffer_.rowRange(c * target_width, (c + 1) * target_width);
//         u8_chan.convertTo(chan_roi, CV_32F, 1.0 / 255.0);
//     }

//     // Shallow reference — no data copy.
//     out_chw_img = ld_chw_buffer_;
// }


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
    if (!original_frame_)
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
        original_frame_->ClearRegions(); // Reset per-frame RC outputs
        original_frame_->ClearLandmarks(); // Reset per-frame LD outputs
    }
    
    img_buff_ = original_frame_->GetImg();
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
    static constexpr float RC_THRESHOLD = 0.5f;
    for (uint8_t i = 0; i < RC_NUM_CLASSES; i++)
    {
        if (host_output[i] > RC_THRESHOLD)
        {
            spdlog::info("RC: class {} score {:.3f} above threshold", GetRegionString(static_cast<RegionID>(i)), host_output[i]);
            original_frame_->AddRegion(static_cast<RegionID>(i), host_output[i]);
        }
    }

    original_frame_->SetProcessingStage(ProcessingStage::RCNeted);

    return EC::OK;
}

EC Orchestrator::ExecLDInferenceTRT()
{
    if (!original_frame_)
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
        original_frame_->ClearLandmarks(); // Reset per-frame LD outputs
    }

    spdlog::info("Regions detected in the frame: {}", original_frame_->GetRegionIDs().size());

    const auto& selected_region_ids = original_frame_->GetRegionIDs();
    auto release_host_preprocess_scratch = [this]()
    {
        // MEMORY: Aggressive per-frame host scratch release (drops large OpenCV buffers each frame).
        ld_chw_buffer_.release();
        ld_letterboxed_u8_.release();
        img_buff_.release();
    };

    if (selected_region_ids.empty())
    {
        spdlog::warn("No regions detected by RCNet. Skipping LD inference.");
        // Evict all live engines — RC sees nothing, free all GPU memory.
        if (!ld_nets_.empty())
        {
            spdlog::info("Evicting {} live LDNet engine(s) — no regions selected.", ld_nets_.size());
            ld_nets_.clear();
        }
        release_host_preprocess_scratch();
        return EC::OK;
    }

    // Evict engines for regions no longer selected by RC.
    for (auto it = ld_nets_.begin(); it != ld_nets_.end(); )
    {
        if (std::find(selected_region_ids.begin(), selected_region_ids.end(), it->first) == selected_region_ids.end())
        {
            spdlog::info("Evicting LDNet for deselected region: {}", GetRegionString(it->first));
            it = ld_nets_.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Lazily load engines for newly selected regions.
    std::vector<LDNet*> ld_nets;
    std::vector<RegionID> region_ids;
    for (const auto& region_id : selected_region_ids)
    {
        if (ld_assets_.find(region_id) == ld_assets_.end())
        {
            spdlog::error("No LD assets registered for region: {}", GetRegionString(region_id));
            continue;
        }

        // Load engine on first selection of this region.
        if (ld_nets_.find(region_id) == ld_nets_.end())
        {
            const auto& assets = ld_assets_.at(region_id);
            spdlog::info("Lazy-loading LDNet for region: {}", GetRegionString(region_id));
            ld_nets_.try_emplace(region_id, region_id, assets.csv_path);
            EC load_status = ld_nets_.at(region_id).LoadEngine(assets.trt_path);
            if (load_status != EC::OK)
            {
                spdlog::error("Failed to load LDNet engine for region: {}", GetRegionString(region_id));
                ld_nets_.erase(region_id);
                continue;
            }
            spdlog::info("LDNet loaded for region: {}", GetRegionString(region_id));
        }

        ld_nets.push_back(&ld_nets_.at(region_id));
        region_ids.push_back(region_id);
        spdlog::info("Selected LDNet for region: {}", GetRegionString(region_id));
    }

    if (ld_nets.empty())
    {
        spdlog::warn("No initialized LDNet matches RC output regions. Skipping LD inference for this frame.");
        release_host_preprocess_scratch();
        return EC::OK;
    }

    // LD inference
    cv::Mat ld_chw_img;
    LDPreprocessImg(img_buff_, ld_chw_img);
    for (int idx = 0; idx < (int)ld_nets.size(); idx++)
    {
        spdlog::info("Running LD inference for region: {}", GetRegionString(region_ids[idx]));

        // MEMORY: Use LDNet-owned output buffer (raw mode) and avoid per-frame large host allocations.
        float* raw_out = ld_nets[idx]->GetOutputBuffer();
        EC ld_inference_status = ld_nets[idx]->Infer(ld_chw_img.data, raw_out);
        if (ld_inference_status != EC::OK)
        {
            spdlog::error("LDNet inference failed for region: {}", GetRegionString(region_ids[idx]));
            LogError(ld_inference_status);
            ld_nets[idx]->ReleaseScratchBuffers();
            release_host_preprocess_scratch();
            return ld_inference_status;
        }

        // MEMORY: Raw output buffer may be lazily allocated during Infer().
        raw_out = ld_nets[idx]->GetOutputBuffer();
        EC postprocess_status = ld_nets[idx]->PostprocessOutput(raw_out, original_frame_);
        // MEMORY: Aggressive per-frame GPU + host scratch release per LD engine.
        ld_nets[idx]->ReleaseScratchBuffers();
        if (postprocess_status != EC::OK)
        {
            spdlog::error("LDNet postprocess failed for region: {}", GetRegionString(region_ids[idx]));
            LogError(postprocess_status);
            release_host_preprocess_scratch();
            return postprocess_status;
        }

        spdlog::info("LDNet inference output post-processed");
    }
    original_frame_->SetProcessingStage(ProcessingStage::LDNeted);
    release_host_preprocess_scratch();
    
    return EC::OK;
}

EC Orchestrator::ExecLDInferenceONNX()
{
    // if (!current_frame_) 
    // {
    //     spdlog::error("No frame to process");
    //     LogError(EC::NN_NO_FRAME_AVAILABLE);
    //     return EC::NN_NO_FRAME_AVAILABLE;
    // }

    // if (!rc_net_.IsInitialized()) 
    // {
    //     spdlog::error("RCNet is not initialized. Cannot perform inference.");
    //     LogError(EC::NN_ENGINE_NOT_INITIALIZED);
    //     return EC::NN_ENGINE_NOT_INITIALIZED;
    // }

    // if (num_inference_performed_on_current_frame_ > 0) 
    // {
    //     spdlog::warn("Inference already performed on the current frame. This will overwrite.");
    //     original_frame_->ClearLandmarks(); // Reset per-frame LD outputs
    // }
        

    // if (original_frame_->GetRegionIDs().empty())
    // {
    //     spdlog::warn("No regions detected by RCNet. Skipping LD inference.");
    //     return EC::OK;
    // }

    // spdlog::info("Regions detected in the frame: {}", original_frame_->GetRegionIDs().size());
    // std::string onnx_path;
    // std::string csv_path;
    // LDNet* ld_net;
    // std::vector<RegionID> region_ids;

    // // Pre-process the image for LD
    // cv::Mat ld_chw_img;
    // int output_size;

    // Size size(inpWidth, inpHeight);
    // Image2BlobParams imgParams(
    //     scale,
    //     size,
    //     mean,
    //     swapRB,
    //     CV_32F,
    //     DNN_LAYOUT_NCHW,
    //     paddingMode); // , paddingValue);

    // Image2BlobParams paramNet;
    // paramNet.scalefactor = scale;
    // paramNet.size = size;
    // paramNet.mean = mean;
    // paramNet.swapRB = swapRB;
    // paramNet.paddingmode = paddingMode;
    // cv::dnn::Net net;
    // int backend = 0; // automatically, opencv implementation or cuda?
    // int target = 0; // 0: cpu, 6: cuda, 7: cuda fp16

    // std::vector<Mat> outs;
    // std::vector<int> keep_classIds;
    // std::vector<float> keep_confidences;
    // std::vector<Rect2d> keep_boxes;
    // std::vector<Rect> boxes;
    // spdlog::info("Forward buffers initialized");

    // Mat inp;
    // //![preprocess_call_func]
    // inp = blobFromImageWithParams(img, imgParams);
    // spdlog::info("Blob created from image");

    // for(const auto& region_id : original_frame_->GetRegionIDs())
    // {
    //     if (ld_nets_.find(region_id) == ld_nets_.end())
    //     {
    //         spdlog::error("No LDNet found for region: {}", GetRegionString(region_id));
    //         continue;
    //     }
    //     if (!ld_nets_.at(region_id).IsInitialized())
    //     {
    //         spdlog::error("LDNet for region {} is not initialized.", GetRegionString(region_id));
    //         continue;
    //     }
    //     ld_nets.push_back(&ld_nets_.at(region_id));
    //     region_ids.push_back(region_id);
    //     spdlog::info("Selected LDNet for region: {}", GetRegionString(region_id));

    //     // Init the Net
    //     net = readNet(ld_onnx_file_path);
    //     spdlog::info("Model loaded from: {}", ld_onnx_file_path);
    //     net.setPreferableBackend(backend);
    //     net.setPreferableTarget(target);

    //     net.setInput(inp);
    //     spdlog::info("Input set to network");
        
    //     net.forward(outs, net.getUnconnectedOutLayersNames());
    //     spdlog::info("Forward pass completed, outputs: {}", outs.size());


        // Non-max suppression and populate landmarks
        // yoloPostProcessing(
        //     outs, keep_classIds, keep_confidences, keep_boxes,
        //     confThreshold, nmsThreshold,
        //     yolo_model,
        //     nc);
        // spdlog::info("Post-processing completed, detections: {}", keep_boxes.size());
        // for (auto box : keep_boxes)
        // {
        //     boxes.push_back(Rect(cvFloor(box.x), cvFloor(box.y), cvFloor(box.width - box.x), cvFloor(box.height - box.y)));
        // }
        // for (auto& b : boxes) {
        //     b = scaleBoxBackLetterbox(b, img.size(), size);
        // }

        // Store results in original frame
    // }

    // original_frame_->SetProcessingStage(ProcessingStage::LDNeted);


    // preprocess image for LD

    return EC::OK;
}


EC Orchestrator::ExecFullInference()
{
    // Run Region Classification inference
    EC rc_ec = ExecRCInference();

    if (rc_ec != EC::OK)  return rc_ec;
    
    // Run Landmark Detection inference
    EC ld_ec = ExecLDInferenceTRT();
    num_inference_performed_on_current_frame_++;

    return ld_ec;
}


} // namespace Inference