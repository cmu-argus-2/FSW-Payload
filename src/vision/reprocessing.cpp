#include "vision/reprocessing.hpp"
#include "core/data_handling.hpp"
#include "spdlog/spdlog.h"

#include <filesystem>
#include <fstream>
#include <memory>

namespace fs = std::filesystem;

namespace {

// Parse timestamp and cam_id from a raw image filename: raw_<timestamp>_<cam_id>.{ext}
// Returns false if the filename does not match the expected pattern.
bool ParseRawImageFilename(const fs::path& path, uint64_t& timestamp_out, int& cam_id_out)
{
    const std::string stem = path.stem().string();
    const auto first_us = stem.find('_');
    const auto last_us  = stem.rfind('_');
    if (first_us == std::string::npos || last_us == std::string::npos || first_us == last_us)
        return false;
    if (stem.substr(0, first_us) != "raw")
        return false;
    try {
        timestamp_out = std::stoull(stem.substr(first_us + 1, last_us - first_us - 1));
        cam_id_out    = std::stoi(stem.substr(last_us + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// Load a frame from disk (image pixels + JSON metadata) into frame_out.
// Returns false if either the image or metadata cannot be loaded.
bool LoadFrameFromFolder(uint64_t timestamp, int cam_id,
                         const std::string& folder, Frame& frame_out)
{
    if (!DH::ReadImageFromDisk(timestamp, cam_id, frame_out, folder))
        return false;

    Json metadata = DH::LoadFrameMetadataFromDisk(timestamp, cam_id, folder);
    if (!metadata.empty())
        frame_out.fromJson(metadata);

    return true;
}

// Run the full processing pipeline on frame_ptr up to target_stage.
EC RunPipeline(std::shared_ptr<Frame> frame_ptr, InferenceManager& im,
               ProcessingStage target)
{
    frame_ptr->RunPrefiltering();

    if (target == ProcessingStage::Prefiltered)
        return EC::OK;

    if (frame_ptr->GetImageState() < ImageState::Earth)
    {
        SPDLOG_INFO("Reprocessing: frame ({}, {}) rejected by prefiltering",
                    frame_ptr->GetCamID(), frame_ptr->GetTimestamp());
        return EC::OK;
    }

    return im.ProcessFrame(frame_ptr, target);
}

} // namespace

namespace Reprocessing {

EC Dataset(::Dataset& dataset, InferenceManager& im, ProcessingStage target, bool overwrite)
{
    const std::string folder  = dataset.GetFolderPath();
    const int    rc_ver       = im.GetRCVersion();
    const int    ld_ver       = im.GetLDVersion();
    const LDNetConfig& ld_cfg = im.GetLDNetConfig();

    if (!fs::is_directory(folder))
    {
        SPDLOG_ERROR("Reprocessing::Dataset: folder does not exist: {}", folder);
        return EC::FILE_DOES_NOT_EXIST;
    }

    EC result = EC::OK;
    int processed = 0, skipped = 0, failed = 0;
    std::vector<std::tuple<uint8_t, uint64_t>> discovered_ids;

    for (const auto& entry : fs::directory_iterator(folder))
    {
        if (!entry.is_regular_file()) continue;

        const fs::path& p = entry.path();
        const std::string ext = p.extension().string();
        if (ext != ".jpg" && ext != ".png") continue;

        uint64_t timestamp;
        int cam_id;
        if (!ParseRawImageFilename(p, timestamp, cam_id)) continue;

        discovered_ids.emplace_back(static_cast<uint8_t>(cam_id), timestamp);

        // Load metadata first; only decode the JPEG when reprocessing is actually needed.
        Frame frame;
        Json metadata = DH::LoadFrameMetadataFromDisk(timestamp, cam_id, folder);
        if (!metadata.empty())
            frame.fromJson(metadata);

        if (!frame.ShouldReprocess(target, overwrite, rc_ver, ld_ver, ld_cfg))
        {
            SPDLOG_INFO("Reprocessing::Dataset: skipping frame ({}, {}) — up-to-date",
                        cam_id, timestamp);
            ++skipped;
            continue;
        }

        if (!DH::ReadImageFromDisk(timestamp, cam_id, frame, folder))
        {
            SPDLOG_ERROR("Reprocessing::Dataset: failed to load image for frame ({}, {})", cam_id, timestamp);
            ++failed;
            result = EC::FILE_NOT_FOUND;
            continue;
        }

        frame.ResetProcessing();
        auto frame_ptr = std::make_shared<Frame>(frame);

        EC status = RunPipeline(frame_ptr, im, target);
        if (status != EC::OK)
        {
            SPDLOG_ERROR("Reprocessing::Dataset: pipeline failed for frame ({}, {}): error {}",
                         cam_id, timestamp, to_uint8(status));
            ++failed;
            result = status;
            continue;
        }

        DH::StoreFrameMetadataToDisk(*frame_ptr, folder);
        ++processed;
        SPDLOG_INFO("Reprocessing::Dataset: reprocessed frame ({}, {})", cam_id, timestamp);
    }

    SPDLOG_INFO("Reprocessing::Dataset: done — processed={}, skipped={}, failed={}",
                processed, skipped, failed);

    // Update dataset metadata and write dataset.json to the folder.
    dataset.SetTargetProcessingStage(target);
    dataset.AddStoredFrameIDs(discovered_ids);
    const std::string json_path = folder + "dataset.json";
    std::ofstream ofs(json_path, std::ios::out | std::ios::trunc);
    if (ofs.is_open())
    {
        ofs << dataset.toJson().dump(1, '\t');
        SPDLOG_INFO("Reprocessing::Dataset: wrote {}", json_path);
    }
    else
    {
        SPDLOG_ERROR("Reprocessing::Dataset: failed to write {}", json_path);
    }


    return result;
}

EC Image(const std::string& raw_image_path, InferenceManager& im,
         ProcessingStage target, bool overwrite)
{
    const fs::path p(raw_image_path);
    if (!fs::is_regular_file(p))
    {
        SPDLOG_ERROR("Reprocessing::Image: file not found: {}", raw_image_path);
        return EC::FILE_DOES_NOT_EXIST;
    }

    uint64_t timestamp;
    int cam_id;
    if (!ParseRawImageFilename(p, timestamp, cam_id))
    {
        SPDLOG_ERROR("Reprocessing::Image: filename does not match raw_<ts>_<cam>.{{jpg|png}}: {}",
                     raw_image_path);
        return EC::INVALID_COMMAND_ARGUMENTS;
    }

    const std::string folder  = p.parent_path().string();
    const int    rc_ver       = im.GetRCVersion();
    const int    ld_ver       = im.GetLDVersion();
    const LDNetConfig& ld_cfg = im.GetLDNetConfig();

    Frame frame;
    if (!LoadFrameFromFolder(timestamp, cam_id, folder, frame))
    {
        SPDLOG_ERROR("Reprocessing::Image: failed to load frame ({}, {})", cam_id, timestamp);
        return EC::FILE_NOT_FOUND;
    }

    if (!frame.ShouldReprocess(target, overwrite, rc_ver, ld_ver, ld_cfg))
    {
        SPDLOG_INFO("Reprocessing::Image: skipping ({}, {}) — conditions match or overwrite=false",
                    cam_id, timestamp);
        return EC::OK;
    }

    frame.ResetProcessing();
    auto frame_ptr = std::make_shared<Frame>(frame);

    EC status = RunPipeline(frame_ptr, im, target);
    if (status != EC::OK)
    {
        SPDLOG_ERROR("Reprocessing::Image: pipeline failed for ({}, {}): error {}",
                     cam_id, timestamp, to_uint8(status));
        return status;
    }

    DH::StoreFrameMetadataToDisk(*frame_ptr, folder);
    SPDLOG_INFO("Reprocessing::Image: reprocessed frame ({}, {})", cam_id, timestamp);
    return EC::OK;
}

} // namespace Reprocessing
