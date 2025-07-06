#include "inference/orchestrator.hpp"
#include "spdlog/spdlog.h"


namespace Inference
{

Orchestrator::Orchestrator()
: current_frame_(nullptr)
{

}


void Orchestrator::Initialize()
{
    // Initialize the RCNet runtime
    EC rc_net_status = rc_net_.LoadEngine("path_to_.trt");
    if (rc_net_status != EC::OK) 
    {
        LogError(rc_net_status);
        spdlog::error("Failed to load RC Net engine.");
    }

    // TODO: Initialize the LDs runtimes

}

void Orchestrator::GrabNewImage(std::shared_ptr<Frame> frame)
{
    if (!frame) 
    {
        spdlog::error("Received null frame.");
        return;
    }

    // actual copy here
    current_frame_ = std::make_shared<Frame>(*frame);
    num_inference_performed_on_current_frame_ = 0; // Reset the inference count for the new frame
}


void Orchestrator::ExecFullInference()
{

    if (!current_frame_) 
    {
        spdlog::error("No frame to process");
        return;
    }


    if (num_inference_performed_on_current_frame_ > 0) 
    {
        spdlog::warn("Inference already performed on the current frame. This will overwrite.");
        // TODO: clear
    }

    cv::Mat img = current_frame_->GetImg();

    // Preprocess the image


    // Run the RC net 

    // Populate the RC ID



    // TODO: LD selection

    // TODO: LD inference 

    // TODO: Populate landmarks


    num_inference_performed_on_current_frame_++;
}


} // namespace Inference