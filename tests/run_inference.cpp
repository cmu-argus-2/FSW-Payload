#include <memory>
#include "spdlog/spdlog.h"
#include "vision/frame.hpp"
#include "core/data_handling.hpp"
#include "inference/orchestrator.hpp"
#include "vision/regions.hpp"

/*
    This file is a simple test for the inference orchestrator.
    It grabs a sample image from disk, initializes the orchestrator,
    and runs inference on the image to fill the regions and landmarks.
*/


int main(int argc, char** argv)
{
    if (argc < 4)
    {
        spdlog::error("Usage: {} <path_to_rc_trt_file> <path_to_ld_trt_folder> <path_to_sample_image>", argv[0]);
        return 1;
    }
    std::string rc_trt_file_path = argv[1];
    std::string ld_trt_folder_path = argv[2];
    std::string sample_image_path = argv[3];

    Inference::Orchestrator orchestrator;
    orchestrator.Initialize(rc_trt_file_path, ld_trt_folder_path);

    Frame frame; // empty frame 
    if (!DH::ReadImageFromDisk(sample_image_path, frame))
    {
        spdlog::error("Failed to read image from disk: {}", sample_image_path);
        return 1;
    }

    std::shared_ptr<Frame> frame_ptr = std::make_shared<Frame>(frame);

    orchestrator.GrabNewImage(frame_ptr); 
    spdlog::info("Running inference on the frame...");
    EC status = orchestrator.ExecFullInference();
    if (status != EC::OK)
    {
        spdlog::error("Inference failed with error code: {}", to_uint8(status));
        return 1;
    }
    spdlog::info("Inference completed successfully.");
    spdlog::info("Regions found: {}", frame_ptr->GetRegionIDs().size());

    for (const auto& region_id : frame_ptr->GetRegionIDs())
    {
        spdlog::info("Region ID: {}", GetRegionString(region_id));
    }

    // Landmark Detection results
    std::vector<Landmark> landmarks = frame_ptr->GetLandmarks();
    spdlog::info("Landmarks found: {}", landmarks.size());
    for (const auto& landmark : landmarks)
    {
        spdlog::info("Landmark - Class ID: {}, Region ID: {}, Confidence: {:.3f}, Position: ({:.2f}, {:.2f}), Size: ({:.2f}, {:.2f})",
            landmark.class_id, GetRegionString(landmark.region_id), landmark.confidence, landmark.x, landmark.y, landmark.height, landmark.width);
    }

    return 0;
}