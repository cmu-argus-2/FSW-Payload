#include "vision/reprocessing.hpp"
#include "core/data_handling.hpp"
#include "core/timing.hpp"
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

std::string FrameMetadataPath(const std::string& folder, uint64_t timestamp, int cam_id)
{
    fs::path folder_path(folder.empty() ? "." : folder);
    return (folder_path / ("frame_" + std::to_string(timestamp) + "_" +
                           std::to_string(cam_id) + ".json")).string();
}

// Load a dataset-style frame from disk. Missing metadata is reported separately
// so callers can decide whether to treat the image as a fresh frame.
bool LoadFrameFromFolder(uint64_t timestamp, int cam_id,
                         const std::string& folder, Frame& frame_out,
                         bool& loaded_metadata_out)
{
    loaded_metadata_out = false;
    if (!DH::ReadImageFromDisk(timestamp, cam_id, frame_out, folder))
        return false;

    Json metadata = DH::LoadFrameMetadataFromDisk(timestamp, cam_id, folder);
    if (!metadata.empty()) {
        frame_out.fromJson(metadata);
        loaded_metadata_out = true;
    }

    return true;
}

// Run the full processing pipeline on frame_ptr up to target_stage.
EC RunPipeline(std::shared_ptr<Frame> frame_ptr, InferenceManager& im,
               ProcessingStage target, bool bypass_prefilter_rejection)
{
    frame_ptr->RunPrefiltering();

    if (target == ProcessingStage::Prefiltered)
        return EC::OK;

    if (!bypass_prefilter_rejection && frame_ptr->GetImageState() < ImageState::Earth)
    {
        SPDLOG_INFO("Reprocessing: frame ({}, {}) rejected by prefiltering",
                    frame_ptr->GetCamID(), frame_ptr->GetTimestamp());
        return EC::OK;
    }

    if (bypass_prefilter_rejection && frame_ptr->GetImageState() < ImageState::Earth)
    {
        SPDLOG_INFO("Reprocessing: frame ({}, {}) bypassing prefilter rejection",
                    frame_ptr->GetCamID(), frame_ptr->GetTimestamp());
    }

    return im.ProcessFrame(frame_ptr, target);
}

} // namespace

namespace Reprocessing {

EC Dataset(::Dataset& dataset, InferenceManager& im, ProcessingStage target, bool overwrite,
           bool bypass_prefilter_rejection)
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

        EC status = RunPipeline(frame_ptr, im, target, bypass_prefilter_rejection);
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
         ProcessingStage target, bool overwrite,
         bool bypass_prefilter_rejection,
         std::string* metadata_path_out)
{
    const fs::path p(raw_image_path);
    if (!fs::is_regular_file(p))
    {
        SPDLOG_ERROR("Reprocessing::Image: file not found: {}", raw_image_path);
        return EC::FILE_DOES_NOT_EXIST;
    }

    const std::string folder  = p.parent_path().empty() ? "." : p.parent_path().string();
    const int    rc_ver       = im.GetRCVersion();
    const int    ld_ver       = im.GetLDVersion();
    const LDNetConfig& ld_cfg = im.GetLDNetConfig();

    uint64_t timestamp = 0;
    int cam_id = 0;
    const bool parsed_raw_name = ParseRawImageFilename(p, timestamp, cam_id);

    Frame frame;
    bool loaded_existing_metadata = false;
    bool loaded_frame = false;

    if (parsed_raw_name) {
        loaded_frame = LoadFrameFromFolder(timestamp, cam_id, folder, frame, loaded_existing_metadata);
    }

    if (!loaded_frame || !loaded_existing_metadata)
    {
        if (!parsed_raw_name) {
            timestamp = static_cast<uint64_t>(timing::GetCurrentTimeMs());
            cam_id = 0;
            SPDLOG_INFO("Reprocessing::Image: '{}' is not named raw_<ts>_<cam>; creating fresh frame ({}, {})",
                        raw_image_path, cam_id, timestamp);
        } else {
            SPDLOG_INFO("Reprocessing::Image: no frame metadata found for ({}, {}); creating fresh frame",
                        cam_id, timestamp);
        }

        if (!DH::ReadImageFromDisk(raw_image_path, frame, cam_id, timestamp))
        {
            SPDLOG_ERROR("Reprocessing::Image: failed to load image: {}", raw_image_path);
            return EC::FILE_NOT_FOUND;
        }
        loaded_existing_metadata = false;
    }

    const std::string metadata_path = FrameMetadataPath(folder, frame.GetTimestamp(), frame.GetCamID());
    if (metadata_path_out) *metadata_path_out = metadata_path;

    if (loaded_existing_metadata && !frame.ShouldReprocess(target, overwrite, rc_ver, ld_ver, ld_cfg))
    {
        SPDLOG_INFO("Reprocessing::Image: skipping ({}, {}) — conditions match or overwrite=false",
                    cam_id, timestamp);
        return EC::OK;
    }

    frame.ResetProcessing();
    auto frame_ptr = std::make_shared<Frame>(frame);

    EC status = RunPipeline(frame_ptr, im, target, bypass_prefilter_rejection);
    if (status != EC::OK)
    {
        SPDLOG_ERROR("Reprocessing::Image: pipeline failed for ({}, {}): error {}",
                     cam_id, timestamp, to_uint8(status));
        return status;
    }

    DH::StoreFrameMetadataToDisk(*frame_ptr, folder);
    if (metadata_path_out) {
        *metadata_path_out = FrameMetadataPath(folder, frame_ptr->GetTimestamp(), frame_ptr->GetCamID());
    }
    SPDLOG_INFO("Reprocessing::Image: reprocessed frame ({}, {})", frame_ptr->GetCamID(), frame_ptr->GetTimestamp());
    return EC::OK;
}

} // namespace Reprocessing
