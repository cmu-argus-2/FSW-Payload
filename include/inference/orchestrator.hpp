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

    void Initialize(const std::string& rc_engine_path, const std::string& ld_engine_folder_path);  

    void GrabNewImage(std::shared_ptr<Frame> frame);

    EC ExecRCInference();
    EC ExecLDInferenceTRT();
    EC ExecLDInferenceONNX();
    EC ExecFullInference();

    static size_t GetMemorySize(const nvinfer1::Dims& dims, size_t element_size);


private:

    std::shared_ptr<Frame> original_frame_; // Frame being processed and populated
    int num_inference_performed_on_current_frame_ = 0; 
    cv::Mat img_buff_; // Buffer for the current image

    void RCPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img);
    void LDPreprocessImg(const cv::Mat& img, cv::Mat& out_chw_img, int target_width=4608);
    // void PreprocessImgGPU(const cv::Mat& img, cv::Mat& out_chw);

    // These are released after each frame, they are needed for LD preprocessing
    // May need to be changed later 
    cv::Mat ld_letterboxed_u8_; // uint8
    cv::Mat ld_chw_buffer_;     // float32

    RCNet rc_net_;

    // TRT engines are loaded when RC detects each respective region.
    struct LDAssets { std::string trt_path; std::string csv_path; };
    std::map<RegionID, LDAssets> ld_assets_;

    // Live regions/engines — only regions currently selected by RC PER FRAME are kept.
    // Engines for deselected regions are freed to reclaim GPU memory.
    std::map<RegionID, LDNet> ld_nets_;

};

} // namespace Inference

#endif // ORCHESTRATOR_HPP
