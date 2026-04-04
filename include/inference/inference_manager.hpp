#ifndef INFERENCE_MANAGER_HPP
#define INFERENCE_MANAGER_HPP

#include <memory>
#include <mutex>
#include <string>
#include "vision/frame.hpp"
#include "core/errors.hpp"
#include "inference/types.hpp"

#ifdef CUDA_ENABLED
#include <map>
#include <NvInfer.h>
#include "vision/regions.hpp"
namespace Inference { class RCNet; class LDNet; }
#endif

class InferenceManager
{
public:
    InferenceManager();
    ~InferenceManager();

    // Process a frame from its current ProcessingStage to target_stage.
    // The frame's GetProcessingStage() determines the starting point; the caller
    // is responsible for restoring metadata (e.g. regions) before calling if needed.
    // Returns EC::OK immediately if the frame is already at or beyond target_stage.
    // Thread-safe: serializes concurrent callers.
    EC ProcessFrame(std::shared_ptr<Frame> frame_ptr, ProcessingStage target_stage);

    // Release engine memory. Safe to call between collection windows.
    // Blocks until any in-flight inference call completes.
    void FreeEngines();

    // --- Configuration setters ---
    // Must be called before the first ProcessFrame call (or after FreeEngines to reconfigure).
    // Setters store configuration only; engines are loaded lazily on first ProcessFrame.
    EC SetRCNetEnginePath(const std::string& path);
    EC SetLDNetEngineFolderPath(const std::string& path);
    void SetLDNetConfig(NET_QUANTIZATION weight_quant, int input_width, int input_height,
                        bool embedded_nms, bool use_trt_for_ld);

    // Control whether engines are loaded eagerly at first ProcessFrame or on-demand per inference.
    void SetPreloadRCEngine(bool preload) { preload_rc_engine_ = preload; }
    void SetPreloadLDEngines(bool preload) { preload_ld_engines_ = preload; }

private:
    std::mutex mtx_;

    // Configuration (CUDA-free, always present)
    std::string rc_engine_path_ = "./models/V1/trained-rc/effnet_0997acc.trt";
    std::string ld_engine_folder_path_ = "./models/V1/trained-ld";
    LDNetConfig ldnet_config_ = {NET_QUANTIZATION::FP16, 4608, 2592, false, true};
    bool preload_rc_engine_ = true;
    bool preload_ld_engines_ = false;
    // Minimum free GPU bytes required before loading each LD engine
    size_t min_gpu_free_between_loads_ = 256ULL * 1024 * 1024;

#ifdef CUDA_ENABLED
    // State
    bool initialized_ = false;
    std::shared_ptr<Frame> current_frame_;
    int num_rc_inferences_on_current_frame_ = 0;
    int num_ld_inferences_on_current_frame_ = 0;

    std::unique_ptr<Inference::RCNet> rc_net_;
    std::map<RegionID, std::unique_ptr<Inference::LDNet>> ld_nets_;

    // Bind a new frame for the next exec call. Resets per-frame counters.
    // Called internally by ProcessFrame; exposed for unit testing via #define private public.
    void GrabNewImage(std::shared_ptr<Frame> frame);

    // Initialize TRT plugins, build ld_nets_ map, and optionally preload engines.
    // Called once from ProcessFrame under the lock.
    void EnsureInitialized();

    // Engine loading (private — callers use ProcessFrame or FreeEngines)
    EC LoadRCEngine();
    void LoadLDNetEngines();
    EC LoadLDNetEngineForRegion(RegionID region_id);
    void InitializeLDNetRuntimes();

    // Engine freeing (private)
    void FreeRCNet();
    void FreeLDNets();
    void FreeLDNetForRegion(RegionID region_id);

    // Inference execution (private)
    EC ExecRCInference();
    EC ExecLDInference();
    EC ExecFullInference();

    // Image preprocessing (private)
    void RCPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img);
    void LDPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img,
                         int target_width = 4608, int target_height = 2592);

    static size_t GetMemorySize(const nvinfer1::Dims& dims, size_t element_size);
#endif
};

#endif // INFERENCE_MANAGER_HPP
