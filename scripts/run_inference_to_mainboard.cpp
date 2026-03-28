#include <memory>
#include "spdlog/spdlog.h"
#include "vision/frame.hpp"
#include "vision/camera_manager.hpp"
#include "core/data_handling.hpp"
#include "core/timing.hpp"
#include "inference/orchestrator.hpp"
#include "vision/regions.hpp"
#include "configuration.hpp"
#include "communication/comms.hpp"

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        spdlog::error("Usage: {} <path_to_rc_trt_file> <path_to_ld_trt_folder>", argv[0]);
        return 1;
    }

    std::string rc_trt_file_path = argv[1];
    std::string ld_trt_folder_path = argv[2];

    DH::InitializeDataStorage();

    // Initialize cameras
    spdlog::info("Initializing camera test");
    auto config = std::make_unique<Configuration>();
    config->LoadConfiguration("config/config.toml");
    const auto& cam_configs = config->GetCameraConfigs();
    CameraManager cam_manager(cam_configs);

    spdlog::info("Enabling cameras...");
    std::array<bool, NUM_CAMERAS> activated;
    int count = cam_manager.EnableCameras(activated);
    spdlog::info("Cameras enabled: {}", count);

    spdlog::info("Waiting for cameras to stabilize...");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    spdlog::info("Capturing single frame per camera");
    cam_manager.CaptureFrames();

    spdlog::info("Waiting for frame capture to complete...");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::vector<Frame> frames;
    uint8_t nb_frames = cam_manager.CopyFrames(frames);
    spdlog::info("Copied {} frame(s).", nb_frames);

    if (nb_frames == 0)
    {
        spdlog::error("No frames captured.");
        return 1;
    }

    // Disable cameras before inference to free memory
    spdlog::info("Disabling cameras before inference...");
    std::array<bool, NUM_CAMERAS> disabled;
    cam_manager.DisableCameras(disabled);

    // Initialize orchestrator
    Inference::Orchestrator orchestrator;
    orchestrator.Initialize(rc_trt_file_path, ld_trt_folder_path);

    // Run inference on each frame
    for (auto& frame : frames)
    {
        if (frame.GetProcessingStage() == ProcessingStage::NotPrefiltered)
        {
            frame.RunPrefiltering();
        }

        // Save raw image to disk
        DH::StoreFrameToDisk(frame, IMAGES_FOLDER);

        std::shared_ptr<Frame> frame_ptr = std::make_shared<Frame>(frame);
        orchestrator.GrabNewImage(frame_ptr);
        spdlog::info("Running inference on frame from camera {}...", frame.GetCamID());

        EC status = orchestrator.ExecFullInference();
        if (status != EC::OK)
        {
            spdlog::error("Inference failed with error code: {}", to_uint8(status));
            continue;
        }

        spdlog::info("Inference completed successfully.");
        spdlog::info("Regions found: {}", frame_ptr->GetRegionIDs().size());
        for (const auto& region_id : frame_ptr->GetRegionIDs())
        {
            spdlog::info("Region ID: {}", GetRegionString(region_id));
        }

        // Save metadata JSON
        DH::StoreFrameMetadataToDisk(*frame_ptr, IMAGES_FOLDER);
        spdlog::info("Frame metadata JSON saved.");

        // Build JSON path (same naming convention as StoreFrameMetadataToDisk)
        std::string json_path = std::string(IMAGES_FOLDER) + "raw_" + 
                                std::to_string(frame_ptr->GetTimestamp()) + "_" + 
                                std::to_string(frame_ptr->GetCamID()) + ".json";

        // Packetize JSON to binary in comms folder
        std::string bin_path = std::string(COMMS_FOLDER) + "meta_" +
                               std::to_string(frame_ptr->GetTimestamp()) + "_" +
                               std::to_string(frame_ptr->GetCamID()) + ".bin";

        if (!DH::PacketizeJsonFile(json_path, bin_path))
        {
            spdlog::error("Failed to packetize metadata JSON.");
            continue;
        }
        spdlog::info("Packetized binary: {} ({} bytes)", bin_path, DH::GetFileSize(bin_path));

        // Ready for transfer
        FileTransferManager::Reset();
        EC err = FileTransferManager::PopulateMetadata(bin_path);
        if (err != EC::OK)
        {
            spdlog::error("Failed to populate file transfer metadata.");
            continue;
        }

        spdlog::info("File transfer ready. Total packets: {}", FileTransferManager::total_seq_count());

        // Log landmarks
        std::vector<Landmark> landmarks = frame_ptr->GetLandmarks();
        spdlog::info("Landmarks found: {}", landmarks.size());
        for (const auto& landmark : landmarks)
        {
            spdlog::info("Landmark - Class ID: {}, Region ID: {}, Confidence: {:.3f}, Position: ({:.2f}, {:.2f}), Size: ({:.2f}, {:.2f})",
                landmark.class_id, GetRegionString(landmark.region_id), landmark.confidence,
                landmark.x, landmark.y, landmark.height, landmark.width);
        }
    }

    return 0;
}