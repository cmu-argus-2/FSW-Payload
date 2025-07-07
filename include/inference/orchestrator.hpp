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

    void Initialize();  

    void GrabNewImage(std::shared_ptr<Frame> frame);

    void ExecFullInference();


    static size_t GetMemorySize(const nvinfer1::Dims& dims, size_t element_size);
 


private:

    std::shared_ptr<Frame> current_frame_; // Current frame being processed
    int num_inference_performed_on_current_frame_ = 0; 

    void PreprocessImage();


    RCNet rc_net_; 






};

} // namespace Inference

#endif // ORCHESTRATOR_HPP