#ifndef ORCHESTRATOR_HPP
#define ORCHESTRATOR_HPP

#include "inference/runtimes.hpp"
#include "vision/frame.hpp"

namespace Inference
{

class Orchestrator
{

public:
    Orchestrator();
    ~Orchestrator();

    void FreeEngines();
    void FreeRCNet();
    void FreeLDNets();

    void FreeLDNetForRegion(RegionID region_id);

    void InitializeLDNetRuntimes();

    // TODO: Can the below be moved to private
    // Use these to initialize the class or load the engines on demand?
    EC LoadRCEngine();
    void LoadLDNetEngines(); 
    EC LoadLDNetEngineForRegion(RegionID region_id);

    void GrabNewImage(std::shared_ptr<Frame> frame);

    // Setters
    void SetPreloadRCEngine(bool preload) { preload_rc_engine_ = preload; }
    void SetPreloadLDEngines(bool preload) { preload_ld_engines_ = preload; }
    void SetUseTRTForLD(bool use_trt) { ldnet_config.use_trt = use_trt; }
    EC SetRCNetEnginePath(const std::string& path);
    EC SetLDNetEngineFolderPath(const std::string& path);
    void SetLDNetConfig(NET_QUANTIZATION weight_quant, int input_width, int input_height, bool embedded_nms, bool use_trt_for_ld);

    EC ExecRCInference();
    // TODO: The two functions below should be merged
    EC ExecLDInference();
    EC ExecFullInference();

    static size_t GetMemorySize(const nvinfer1::Dims& dims, size_t element_size);

private:

    std::shared_ptr<Frame> original_frame_; // Frame being processed and populated
    int num_rc_inferences_on_current_frame_ = 0;
    int num_ld_inferences_on_current_frame_ = 0;

    bool preload_rc_engine_ = true; // Option to preload RC engine at initialization
    bool preload_ld_engines_ = false; // Option to preload LD engines at initialization

    // Minimum free GPU bytes required before loading each LD engine in LoadLDNetEngines().
    size_t min_gpu_free_between_loads_ = 256ULL * 1024 * 1024;

    std::string ld_engine_folder_path_ = "./models/V1/trained-ld"; // Folder path for LD engines (if loading on demand)
    std::string rc_engine_path_ = "./models/V1/trained-rc/effnet_0997acc.trt"; // File path for RC engine (if loading on demand)
    LDNetConfig ldnet_config = {NET_QUANTIZATION::FP16,4608,2592,false,true};

    void RCPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img);
    void LDPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img, int target_width=4608, int target_height=2592);
    // void PreprocessImgGPU(const cv::Mat& img, cv::Mat& out_chw);

    RCNet rc_net_; 
    // TODO: Option to preload or load on demand for LDNets
    // Runtime vs memory tradeoff
    std::map<RegionID, std::unique_ptr<LDNet>> ld_nets_; // LDNet engines

};

} // namespace Inference

#endif // ORCHESTRATOR_HPP
