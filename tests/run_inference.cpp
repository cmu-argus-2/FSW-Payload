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
    if (argc < 3)
    {
        spdlog::error("Usage: {} <path_to_trt_file> <path_to_sample_image>", argv[0]);
        return 1;
    }
    std::string trt_file_path = argv[1];
    std::string sample_image_path = argv[2];

    Inference::Orchestrator orchestrator;
    orchestrator.Initialize(trt_file_path);

    Frame frame; // empty frame 
    DH::ReadImageFromDisk(sample_image_path, frame); 

    orchestrator.GrabNewImage(std::make_shared<Frame>(frame)); 
    spdlog::info("Running inference on the frame...");
    EC status = orchestrator.ExecFullInference();
    if (status != EC::OK)
    {
        spdlog::error("Inference failed with error code: {}", to_uint8(status));
        return 1;
    }
    spdlog::info("Inference completed successfully.");
    spdlog::info("Regions found: {}", frame.GetRegionIDs().size());

    for (const auto& region_id : frame.GetRegionIDs())
    {
        spdlog::info("Region ID: {}", GetRegionString(region_id));
    }

    return 0;
}