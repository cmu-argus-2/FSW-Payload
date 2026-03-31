#ifndef INFERENCE_MANAGER_HPP
#define INFERENCE_MANAGER_HPP

#include <memory>
#include <mutex>
#include "vision/frame.hpp"
#include "core/errors.hpp"

#ifdef CUDA_ENABLED
namespace Inference { class Orchestrator; }
#endif

class InferenceManager
{
public:
    InferenceManager() = default;
    ~InferenceManager() = default;

    // Process a frame from its current ProcessingStage to target_stage.
    // The frame's GetProcessingStage() determines the starting point; the caller
    // is responsible for restoring metadata (e.g. regions) before calling if needed.
    // Returns EC::OK immediately if the frame is already at or beyond target_stage.
    // Thread-safe: serializes concurrent callers.
    EC ProcessFrame(std::shared_ptr<Frame> frame_ptr, ProcessingStage target_stage);

    // Release engine memory. Safe to call between collection windows.
    // Blocks until any in-flight inference call completes.
    void FreeEngines();

private:
    std::mutex mtx_;

#ifdef CUDA_ENABLED
    std::unique_ptr<Inference::Orchestrator> orchestrator_;
    void EnsureInitialized();
#endif
};

#endif // INFERENCE_MANAGER_HPP
