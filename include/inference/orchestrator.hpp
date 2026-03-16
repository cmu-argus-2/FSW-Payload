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

    std::shared_ptr<Frame> original_frame_; // Original frame from the camera to populate
    std::shared_ptr<Frame> current_frame_; // Current frame being processed
    int num_inference_performed_on_current_frame_ = 0; 
    cv::Mat img_buff_; // Buffer for the current image

    void RCPreprocessImg(cv::Mat img, cv::Mat& out_chw_img);
    void LDPreprocessImg(cv::Mat img, cv::Mat& out_chw_img, int target_width=4608);
    // void PreprocessImgGPU(const cv::Mat& img, cv::Mat& out_chw);

    RCNet rc_net_; 
    
    std::map<RegionID, std::string> ld_net_onnx_engine_paths_; // onnx paths
    std::map<RegionID, LDNet> ld_nets_; // tensorRT engines

};

} // namespace Inference

#endif // ORCHESTRATOR_HPP