#include "inference/inference_manager.hpp"
#include "spdlog/spdlog.h"

#ifdef CUDA_ENABLED

#include "inference/runtimes.hpp"
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace cv::dnn;

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

InferenceManager::InferenceManager()
    : rc_net_(std::make_unique<Inference::RCNet>())
{
}

InferenceManager::~InferenceManager()
{
    FreeEngines();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

EC InferenceManager::SetRCNetEnginePath(const std::string& path)
{
    if (!std::filesystem::exists(path))
    {
        spdlog::error("InferenceManager: RC engine file not found: {}", path);
        return EC::FILE_DOES_NOT_EXIST;
    }
    rc_engine_path_ = path;
    return EC::OK;
}

EC InferenceManager::SetLDNetEngineFolderPath(const std::string& path)
{
    if (!std::filesystem::exists(path))
    {
        spdlog::error("InferenceManager: LD engine folder not found: {}", path);
        return EC::FILE_DOES_NOT_EXIST;
    }
    ld_engine_folder_path_ = path;
    return EC::OK;
}

void InferenceManager::SetLDNetConfig(NET_QUANTIZATION weight_quant, int input_width,
                                      int input_height, bool embedded_nms, bool use_trt_for_ld)
{
    ldnet_config_ = {weight_quant, input_width, input_height, embedded_nms, use_trt_for_ld};
    // If already initialized, rebuild the ld_nets_ map with the new config.
    if (initialized_)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        InitializeLDNetRuntimes();
    }
}

EC InferenceManager::ProcessFrame(std::shared_ptr<Frame> frame_ptr, ProcessingStage target_stage)
{
    if (!frame_ptr)
    {
        SPDLOG_ERROR("InferenceManager::ProcessFrame: null frame");
        return EC::NN_POINTER_NULL;
    }

    if (frame_ptr->GetProcessingStage() >= target_stage)
        return EC::OK;

    std::lock_guard<std::mutex> lock(mtx_);

    // Re-check after acquiring lock — another caller may have advanced the stage
    if (frame_ptr->GetProcessingStage() >= target_stage)
        return EC::OK;

    EnsureInitialized();

    const ProcessingStage current_stage = frame_ptr->GetProcessingStage();
    GrabNewImage(frame_ptr);

    EC status;
    if (current_stage < ProcessingStage::RCNeted && target_stage == ProcessingStage::RCNeted)
    {
        status = ExecRCInference();
    }
    else if (current_stage < ProcessingStage::RCNeted && target_stage == ProcessingStage::LDNeted)
    {
        status = ExecFullInference();
    }
    else // current_stage >= RCNeted, target == LDNeted
    {
        status = ExecLDInference();
    }

    if (status != EC::OK)
    {
        SPDLOG_ERROR("InferenceManager: inference failed for frame ({}, {}): error {}",
                     frame_ptr->GetCamID(), frame_ptr->GetTimestamp(), to_uint8(status));
    }

    return status;
}

void InferenceManager::FreeEngines()
{
    std::lock_guard<std::mutex> lock(mtx_);
    FreeRCNet();
    FreeLDNets();
    current_frame_ = nullptr;
    initialized_ = false;
    SPDLOG_INFO("InferenceManager: engines freed.");
}

// ---------------------------------------------------------------------------
// Private: initialization
// ---------------------------------------------------------------------------

void InferenceManager::EnsureInitialized()
{
    if (initialized_)
        return;

    // Must be called once before any deserializeCudaEngine().
    initLibNvInferPlugins(nullptr, "");

    // Validate paths before trying to load anything.
    if (!std::filesystem::exists(rc_engine_path_))
    {
        SPDLOG_ERROR("InferenceManager: RC engine not found: {}", rc_engine_path_);
    }
    if (!std::filesystem::exists(ld_engine_folder_path_))
    {
        SPDLOG_ERROR("InferenceManager: LD engine folder not found: {}", ld_engine_folder_path_);
    }

    InitializeLDNetRuntimes();

    if (preload_rc_engine_)
    {
        LoadRCEngine();
    }

    initialized_ = true;
    SPDLOG_INFO("InferenceManager initialized.");
}

// ---------------------------------------------------------------------------
// Private: engine loading
// ---------------------------------------------------------------------------

EC InferenceManager::LoadRCEngine()
{
    EC rc_net_status = rc_net_->LoadEngine(rc_engine_path_);
    if (rc_net_status != EC::OK)
    {
        spdlog::error("InferenceManager: failed to load RC engine.");
    }
    return rc_net_status;
}

void InferenceManager::LoadLDNetEngines()
{
    size_t loaded = 0;
    for (const auto& region_id : GetAllRegionIDs())
    {
        size_t gpu_free = 0, gpu_total = 0;
        cudaMemGetInfo(&gpu_free, &gpu_total);
        if (gpu_free < min_gpu_free_between_loads_)
        {
            spdlog::warn("InferenceManager::LoadLDNetEngines: stopping early — only {} MiB free "
                         "(minimum {} MiB). {} engine(s) loaded so far.",
                         gpu_free >> 20, min_gpu_free_between_loads_ >> 20, loaded);
            break;
        }

        EC status = LoadLDNetEngineForRegion(region_id);
        if (status == EC::OK)
            loaded++;
    }

    if (loaded == 0)
        spdlog::warn("InferenceManager: no LD models loaded from: {}", ld_engine_folder_path_);
    else
        spdlog::info("InferenceManager: loaded {} LD model(s) from: {}", loaded, ld_engine_folder_path_);
}

EC InferenceManager::LoadLDNetEngineForRegion(RegionID region_id)
{
    std::string region_str = std::string(GetRegionString(region_id));
    std::string engine_path = ld_engine_folder_path_ + "/" + region_str + "/" +
                              region_str + ldnet_config_.GetFileNameAppendix();

    if (!std::filesystem::exists(engine_path))
    {
        spdlog::error("InferenceManager: LD engine not found for region {}: {}",
                      region_str, engine_path);
        return EC::NN_FAILED_TO_OPEN_ENGINE_FILE;
    }

    if (ld_nets_.find(region_id) == ld_nets_.end())
    {
        spdlog::error("InferenceManager: no LDNet runtime for region {}. "
                      "Call InitializeLDNetRuntimes first.", region_str);
        return EC::NN_ENGINE_NOT_INITIALIZED;
    }

    EC status = ld_nets_.at(region_id)->LoadEngine(engine_path);
    if (status != EC::OK)
        spdlog::error("InferenceManager: failed to load LD engine for region: {}", region_str);
    else
        spdlog::info("InferenceManager: LDNet loaded for region: {}", region_str);

    return status;
}

void InferenceManager::InitializeLDNetRuntimes()
{
    FreeLDNets();
    ld_nets_.clear();

    for (const auto& region_id : GetAllRegionIDs())
    {
        std::string region_str = std::string(GetRegionString(region_id));
        std::string csv_path = ld_engine_folder_path_ + "/" + region_str + "/bounding_boxes.csv";
        std::string engine_path = ld_engine_folder_path_ + "/" + region_str + "/" +
                                  region_str + ldnet_config_.GetFileNameAppendix();

        if (!std::filesystem::exists(engine_path) || !std::filesystem::exists(csv_path))
        {
            spdlog::error("InferenceManager: cannot init LDNet for region {}. "
                          "Engine exists: {}, CSV exists: {}",
                          region_str,
                          std::filesystem::exists(engine_path),
                          std::filesystem::exists(csv_path));
            continue;
        }

        ld_nets_[region_id] = std::make_unique<Inference::LDNet>(region_id, csv_path);
        spdlog::info("InferenceManager: LDNet runtime created for region {}: engine={}, csv={}",
                     region_str, engine_path, csv_path);
    }

    if (preload_ld_engines_)
    {
        LoadLDNetEngines();
    }
}

// ---------------------------------------------------------------------------
// Private: engine freeing
// ---------------------------------------------------------------------------

void InferenceManager::FreeRCNet()
{
    if (rc_net_)
        rc_net_->Free();
}

void InferenceManager::FreeLDNets()
{
    for (auto& [region_id, ld_net] : ld_nets_)
    {
        if (ld_net)
            ld_net->Free();
    }
}

void InferenceManager::FreeLDNetForRegion(RegionID region_id)
{
    auto it = ld_nets_.find(region_id);
    if (it != ld_nets_.end() && it->second)
    {
        it->second->Free();
        spdlog::info("InferenceManager: freed LDNet for region: {}", GetRegionString(region_id));
    }
}

// ---------------------------------------------------------------------------
// Private: preprocessing
// ---------------------------------------------------------------------------

void InferenceManager::RCPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img)
{
    if (img.empty())
    {
        spdlog::error("InferenceManager::RCPreprocessImg: input image is empty.");
        out_chw_img.release();
        return;
    }

    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

    const cv::Scalar scalefactor(
        1.0 / (255.0 * 0.229),
        1.0 / (255.0 * 0.224),
        1.0 / (255.0 * 0.225),
        0.0
    );
    const cv::Scalar mean(
        255.0 * 0.485,
        255.0 * 0.456,
        255.0 * 0.406,
        0.0
    );

    cv::dnn::Image2BlobParams params(
        scalefactor,
        Size(224, 224),
        mean,
        false,
        CV_32F,
        DNN_LAYOUT_NCHW
    );

    blobFromImageWithParams(rgb_img, out_chw_img, params);
}

void InferenceManager::LDPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img,
                                       int target_width, int target_height)
{
    Scalar scale = Scalar(1.0/255.0, 1.0/255.0, 1.0/255.0);
    Scalar mean  = Scalar(0.0, 0.0, 0.0);

    Image2BlobParams imgParams(
        scale,
        Size(target_width, target_height),
        mean,
        true,                  // swapRB
        CV_32F,
        DNN_LAYOUT_NCHW,
        DNN_PMODE_LETTERBOX
    );

    out_chw_img = blobFromImageWithParams(img, imgParams);
}

// ---------------------------------------------------------------------------
// Private: frame binding
// ---------------------------------------------------------------------------

void InferenceManager::GrabNewImage(std::shared_ptr<Frame> frame)
{
    if (!frame)
    {
        spdlog::error("InferenceManager::GrabNewImage: null frame.");
        return;
    }
    current_frame_ = frame;
    num_rc_inferences_on_current_frame_ = 0;
    num_ld_inferences_on_current_frame_ = 0;
}

// ---------------------------------------------------------------------------
// Private: inference execution
// ---------------------------------------------------------------------------

EC InferenceManager::ExecRCInference()
{
    if (!current_frame_)
    {
        spdlog::error("InferenceManager::ExecRCInference: no frame set");
        LogError(EC::NN_NO_FRAME_AVAILABLE);
        return EC::NN_NO_FRAME_AVAILABLE;
    }

    if (!preload_rc_engine_ && !rc_net_->IsInitialized())
    {
        LoadRCEngine();
    }

    if (!rc_net_->IsInitialized())
    {
        EC status = LoadRCEngine();
        if (status != EC::OK)
        {
            spdlog::error("InferenceManager: RCNet not initialized, cannot perform inference.");
            LogError(status);
            return status;
        }
    }

    if (num_rc_inferences_on_current_frame_ > 0)
    {
        spdlog::warn("InferenceManager: RC inference already done on this frame. Overwriting.");
        current_frame_->ClearRegions();
        current_frame_->ClearLandmarks();
        num_ld_inferences_on_current_frame_ = 0;
    }

    cv::Mat img = current_frame_->GetImg();
    cv::Mat chw_img;
    RCPreprocessImg(img, chw_img);
    spdlog::info("InferenceManager: image preprocessed, shape {}x{}x{}",
                 chw_img.rows, chw_img.cols, chw_img.channels());

    int rc_num_classes = rc_net_->GetNumClasses();
    auto host_output = std::unique_ptr<float[]>{new float[rc_num_classes]};
    EC status = rc_net_->Infer(chw_img.data, host_output.get());
    if (status != EC::OK)
    {
        spdlog::error("InferenceManager: RCNet inference failed: {}", to_uint8(status));
        LogError(status);
        if (!preload_rc_engine_) FreeRCNet();
        return status;
    }

    static constexpr float RC_THRESHOLD = 0.5f;
    for (uint8_t i = 0; i < rc_num_classes; i++)
    {
        if (host_output[i] > RC_THRESHOLD)
        {
            spdlog::info("InferenceManager: RC class {} score {:.3f}",
                         GetRegionString(static_cast<RegionID>(i)), host_output[i]);
            current_frame_->AddRegion(static_cast<RegionID>(i), host_output[i]);
        }
    }

    current_frame_->SetProcessingStage(ProcessingStage::RCNeted);
    num_rc_inferences_on_current_frame_++;

    if (!preload_rc_engine_) FreeRCNet();

    return EC::OK;
}

EC InferenceManager::ExecLDInference()
{
    if (!current_frame_)
    {
        spdlog::error("InferenceManager::ExecLDInference: no frame set");
        LogError(EC::NN_NO_FRAME_AVAILABLE);
        return EC::NN_NO_FRAME_AVAILABLE;
    }

    if (num_ld_inferences_on_current_frame_ > 0)
    {
        spdlog::warn("InferenceManager: LD inference already done on this frame. Overwriting.");
        current_frame_->ClearLandmarks();
    }

    if (current_frame_->GetRegionIDs().empty())
    {
        spdlog::warn("InferenceManager: no regions detected. Skipping LD inference.");
        return EC::OK;
    }

    spdlog::info("InferenceManager: running LD on {} region(s).",
                 current_frame_->GetRegionIDs().size());

    cv::Mat img = current_frame_->GetImg();

    std::vector<EC> ld_load_statuses;
    std::vector<EC> ld_net_statuses;

    for (const auto& region_id : current_frame_->GetRegionIDs())
    {
        if (ld_nets_.find(region_id) == ld_nets_.end())
        {
            spdlog::error("InferenceManager: no LDNet for region: {}", GetRegionString(region_id));
            continue;
        }

        EC ld_status = EC::OK;

        if (!ld_nets_.at(region_id)->IsInitialized())
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            ld_status = LoadLDNetEngineForRegion(region_id);
            auto t1 = std::chrono::high_resolution_clock::now();
            spdlog::info("InferenceManager: LD engine load for region {} took {} ms",
                         GetRegionString(region_id),
                         std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

            if (ld_status != EC::OK)
            {
                spdlog::error("InferenceManager: failed to load LD engine for region {}. Skipping.",
                              GetRegionString(region_id));
                ld_load_statuses.push_back(ld_status);
                continue;
            }
            if (!ld_nets_.at(region_id)->IsInitialized())
            {
                spdlog::error("InferenceManager: LDNet for region {} still not initialized.",
                              GetRegionString(region_id));
                ld_load_statuses.push_back(EC::NN_ENGINE_NOT_INITIALIZED);
                continue;
            }
        }
        ld_load_statuses.push_back(ld_status);

        cv::Mat ld_chw_img;
        auto t0 = std::chrono::high_resolution_clock::now();
        LDPreprocessImg(img, ld_chw_img,
                        ld_nets_.at(region_id)->GetInputWidth(),
                        ld_nets_.at(region_id)->GetInputHeight());
        auto t1 = std::chrono::high_resolution_clock::now();
        spdlog::info("InferenceManager: LD preprocess for region {} took {} ms",
                     GetRegionString(region_id),
                     std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

        spdlog::info("InferenceManager: running LD inference for region: {}",
                     GetRegionString(region_id));

        int output_size = ld_nets_[region_id]->GetOutputSize();
        t0 = std::chrono::high_resolution_clock::now();

        if (ld_nets_[region_id]->IsTRT())
        {
            float* raw_out = ld_nets_[region_id]->GetOutputBuffer();
            ld_status = ld_nets_[region_id]->Infer(ld_chw_img.data, raw_out);

            t1 = std::chrono::high_resolution_clock::now();
            spdlog::info("InferenceManager: LD inference took {} ms",
                         std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

            if (ld_status != EC::OK)
            {
                spdlog::error("InferenceManager: LDNet inference failed: {}", to_uint8(ld_status));
                LogError(ld_status);
                ld_nets_[region_id]->ReleaseScratchBuffers();
                ld_net_statuses.push_back(ld_status);
                continue;
            }
            spdlog::info("InferenceManager: LDNet inference OK, output size: {}", output_size);

            raw_out = ld_nets_[region_id]->GetOutputBuffer();
            if (!raw_out)
            {
                spdlog::error("InferenceManager: LDNet output buffer is null after inference.");
                LogError(EC::NN_POINTER_NULL);
                ld_nets_[region_id]->ReleaseScratchBuffers();
                ld_net_statuses.push_back(EC::NN_POINTER_NULL);
                continue;
            }

            int sizes[3] = {1,
                            ld_nets_[region_id]->GetNumLandmarks() + 4,
                            ld_nets_[region_id]->GetNumYoloBoxes()};
            cv::Mat output_matrix(3, sizes, CV_32F, raw_out);

            t0 = std::chrono::high_resolution_clock::now();
            EC postprocess_status = ld_nets_[region_id]->PostprocessOutput(output_matrix, current_frame_);
            ld_nets_[region_id]->ReleaseScratchBuffers();
            t1 = std::chrono::high_resolution_clock::now();
            spdlog::info("InferenceManager: LD post-process took {} ms",
                         std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

            if (postprocess_status != EC::OK)
            {
                spdlog::error("InferenceManager: LDNet postprocess failed: {}",
                              to_uint8(postprocess_status));
                LogError(postprocess_status);
                ld_net_statuses.push_back(postprocess_status);
                continue;
            }
            ld_net_statuses.push_back(EC::OK);
        }
        else
        {
            std::vector<cv::Mat> outs;
            ld_status = ld_nets_.at(region_id)->Infer(ld_chw_img, outs);

            t1 = std::chrono::high_resolution_clock::now();
            spdlog::info("InferenceManager: LD inference took {} ms",
                         std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

            if (ld_status != EC::OK)
            {
                spdlog::error("InferenceManager: LDNet ONNX inference failed: {}",
                              to_uint8(ld_status));
                LogError(ld_status);
                ld_net_statuses.push_back(ld_status);
                continue;
            }
            spdlog::info("InferenceManager: LDNet ONNX inference OK, output size: {}", output_size);

            t0 = std::chrono::high_resolution_clock::now();
            EC postprocess_status = ld_nets_[region_id]->PostprocessOutput(outs[0], current_frame_);
            t1 = std::chrono::high_resolution_clock::now();
            spdlog::info("InferenceManager: LD post-process took {} ms",
                         std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

            if (postprocess_status != EC::OK)
            {
                spdlog::error("InferenceManager: LDNet ONNX postprocess failed: {}",
                              to_uint8(postprocess_status));
                LogError(postprocess_status);
                ld_net_statuses.push_back(postprocess_status);
                continue;
            }
            ld_net_statuses.push_back(EC::OK);
        }

        if (!preload_ld_engines_) FreeLDNetForRegion(region_id);
    }

    if (!ld_load_statuses.empty() &&
        std::none_of(ld_load_statuses.begin(), ld_load_statuses.end(),
                     [](EC ec){ return ec == EC::OK; }))
    {
        LogError(ld_load_statuses.front());
        return ld_load_statuses.front();
    }

    if (!ld_net_statuses.empty() &&
        std::none_of(ld_net_statuses.begin(), ld_net_statuses.end(),
                     [](EC ec){ return ec == EC::OK; }))
    {
        LogError(ld_net_statuses.front());
        return ld_net_statuses.front();
    }

    current_frame_->SetProcessingStage(ProcessingStage::LDNeted);
    num_ld_inferences_on_current_frame_++;

    return EC::OK;
}

EC InferenceManager::ExecFullInference()
{
    EC rc_ec = ExecRCInference();
    if (rc_ec != EC::OK) return rc_ec;
    return ExecLDInference();
}

size_t InferenceManager::GetMemorySize(const nvinfer1::Dims& dims, size_t element_size)
{
    size_t size = element_size;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            spdlog::error("InferenceManager::GetMemorySize: dynamic dimension at index {} (value {}).",
                          i, dims.d[i]);
            return 0;
        }
        size *= static_cast<size_t>(dims.d[i]);
    }
    return size;
}

#else // CUDA_ENABLED not defined

InferenceManager::InferenceManager() = default;
InferenceManager::~InferenceManager() = default;

EC InferenceManager::SetRCNetEnginePath(const std::string& path)
{
    rc_engine_path_ = path; // No filesystem check without CUDA
    return EC::OK;
}

EC InferenceManager::SetLDNetEngineFolderPath(const std::string& path)
{
    ld_engine_folder_path_ = path; // No filesystem check without CUDA
    return EC::OK;
}

void InferenceManager::SetLDNetConfig(NET_QUANTIZATION weight_quant, int input_width,
                                      int input_height, bool embedded_nms, bool use_trt_for_ld)
{
    ldnet_config_ = {weight_quant, input_width, input_height, embedded_nms, use_trt_for_ld};
}

EC InferenceManager::ProcessFrame(std::shared_ptr<Frame> /*frame_ptr*/, ProcessingStage /*target_stage*/)
{
    SPDLOG_WARN("InferenceManager::ProcessFrame: inference not available (CUDA disabled)");
    return EC::NN_ENGINE_NOT_INITIALIZED;
}

void InferenceManager::FreeEngines()
{
    // No-op without CUDA
}

#endif // CUDA_ENABLED
