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
    // ~Orchestrator();

    void Initialize(const std::string& rc_engine_path);  

    void GrabNewImage(std::shared_ptr<Frame> frame);

    EC ExecFullInference();

    static size_t GetMemorySize(const nvinfer1::Dims& dims, size_t element_size);


private:

    std::shared_ptr<Frame> current_frame_; // Current frame being processed
    int num_inference_performed_on_current_frame_ = 0; 
    cv::Mat img_buff_; // Buffer for the current image

    void PreprocessImg(cv::Mat img, cv::Mat& out_chw_img);
    // void PreprocessImgGPU(const cv::Mat& img, cv::Mat& out_chw);

    RCNet rc_net_; 

};

} // namespace Inference

#endif // ORCHESTRATOR_HPP