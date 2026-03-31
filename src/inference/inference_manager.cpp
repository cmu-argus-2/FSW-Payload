#include "inference/inference_manager.hpp"
#include "spdlog/spdlog.h"

#ifdef CUDA_ENABLED

#include "inference/orchestrator.hpp"

void InferenceManager::EnsureInitialized()
{
    if (!orchestrator_)
    {
        orchestrator_ = std::make_unique<Inference::Orchestrator>();
        SPDLOG_INFO("InferenceManager: orchestrator initialized");
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
    orchestrator_->GrabNewImage(frame_ptr);

    EC status;
    if (current_stage < ProcessingStage::RCNeted && target_stage == ProcessingStage::RCNeted)
    {
        status = orchestrator_->ExecRCInference();
    }
    else if (current_stage < ProcessingStage::RCNeted && target_stage == ProcessingStage::LDNeted)
    {
        status = orchestrator_->ExecFullInference();
    }
    else // current_stage >= RCNeted, target == LDNeted
    {
        status = orchestrator_->ExecLDInference();
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
    if (orchestrator_)
    {
        orchestrator_->FreeEngines();
        orchestrator_.reset();
        SPDLOG_INFO("InferenceManager: engines freed");
    }
}

#else // CUDA_ENABLED not defined

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
