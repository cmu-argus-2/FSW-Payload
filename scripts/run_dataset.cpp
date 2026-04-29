/*
    Test file to replicate what should happen in FSW when the command to start
    dataset collection is received.

    Usage: run_dataset [--config <dataset_config.toml>] [--out <out_path>]

      --config  Path to dataset config TOML. Falls back to the default if not
                provided or the file cannot be opened.
      --out     File to write the generated dataset folder path into. Falls back
                to path.out if not provided or not writable.
*/
#include "spdlog/spdlog.h"
#include "vision/dataset_manager.hpp"
#include "inference/inference_manager.hpp"
#include "core/timing.hpp"
#include "configuration.hpp"
#include <memory>
#include <thread>
#include <array>
#include <fstream>
#include <string>
#include <vector>

#define DATASET_KEY_CMD "CMD"

static constexpr const char* kDefaultDSConfigPath = "config/dataset_config.toml";
static constexpr const char* kDefaultOutPath      = "path.out";

// Returns the value of --flag, or default_val if not present.
static std::string GetFlag(int argc, char** argv, const char* flag, const char* default_val = "")
{
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == flag) return argv[i + 1];
    return default_val;
}

// Returns the provided path if openable, otherwise falls back to kDefaultDSConfigPath.
static std::string ResolveConfigPath(const std::string& path)
{
    std::ifstream f(path);
    if (f.good()) return path;
    if (path != kDefaultDSConfigPath)
        spdlog::warn("Dataset config '{}' not found, using default '{}'", path, kDefaultDSConfigPath);
    return kDefaultDSConfigPath;
}

// Returns the provided path if writable, otherwise falls back to kDefaultOutPath.
static std::string ResolveOutPath(const std::string& path)
{
    if (path == kDefaultOutPath) return path;
    std::ofstream f(path, std::ios::app);
    if (f.is_open()) return path;
    spdlog::warn("Cannot write to '{}', using default output '{}'", path, kDefaultOutPath);
    return kDefaultOutPath;
}

static void WriteResult(const std::string& out_file, const std::string& content)
{
    std::ofstream f(out_file, std::ios::trunc);
    if (!f.is_open()) { spdlog::error("Failed to write result to '{}'", out_file); return; }
    f << content << '\n';
}


int run(int argc, char** argv)
{
    const std::string ds_config_path = ResolveConfigPath(
        GetFlag(argc, argv, "--config", kDefaultDSConfigPath));
    const std::string out_path = ResolveOutPath(
        GetFlag(argc, argv, "--out", kDefaultOutPath));

    std::string config_file_path = "config/config.toml";
    auto config = std::make_unique<Configuration>();
    try {
        config->LoadConfiguration(config_file_path);
    } catch (const toml::parse_error& err) {
        spdlog::error("Parsing configuration file failed: {}", err.description());
        return to_uint8(EC::PLACEHOLDER);
    }
    SPDLOG_INFO("Configuration file {} loaded.", config_file_path);

    DatasetConfig ds_config;
    try {
        toml::table ds_cfg = toml::parse_file(ds_config_path);
        ds_config.maximum_period      = ds_cfg["maximum_period"].value_or(ds_config.maximum_period);
        ds_config.target_frame_nb     = static_cast<uint8_t>(ds_cfg["target_frame_nb"].value_or(uint64_t(ds_config.target_frame_nb)));
        ds_config.capture_mode        = static_cast<CAPTURE_MODE>(ds_cfg["dataset_capture_mode"].value_or(uint64_t(ds_config.capture_mode)));
        ds_config.imu_collection_mode = static_cast<IMU_COLLECTION_MODE>(ds_cfg["imu_collection_mode"].value_or(uint64_t(ds_config.imu_collection_mode)));
        ds_config.image_capture_rate  = static_cast<uint8_t>(ds_cfg["image_capture_rate"].value_or(uint64_t(ds_config.image_capture_rate)));
        ds_config.imu_sample_rate_hz  = static_cast<float>(ds_cfg["imu_sample_rate_hz"].value_or(double(ds_config.imu_sample_rate_hz)));
        ds_config.target_processing_stage = static_cast<ProcessingStage>(ds_cfg["target_processing_stage"].value_or(uint64_t(ds_config.target_processing_stage)));
        if (const auto* arr = ds_cfg["active_cameras"].as_array())
            for (size_t i = 0; i < NUM_CAMERAS && i < arr->size(); ++i)
                if (auto val = (*arr)[i].value<bool>()) ds_config.active_cameras[i] = *val;
    } catch (const toml::parse_error& err) {
        spdlog::error("Failed to parse dataset config {}: {}", ds_config_path, err.description());
        return to_uint8(EC::PLACEHOLDER);
    }

    if (ds_config.maximum_period == 0.0 || ds_config.target_frame_nb == 0)
    {
        spdlog::error("Invalid parameters for dataset collection command");
        return to_uint8(EC::INVALID_COMMAND_ARGUMENTS);
    }

    ds_config.capture_start_time = static_cast<uint64_t>(timing::GetCurrentTimeMs());

    const auto& imu_config = config->GetIMUConfig();
    IMUManager imu_manager(imu_config);

    InferenceManager inference_manager;

    const auto& cam_configs = config->GetCameraConfigs();
    const auto& isp_config  = config->GetCameraISPConfig();
    CameraManager camera_manager(cam_configs, isp_config, inference_manager);

    std::thread imu_thread = std::thread(&IMUManager::RunLoop, &imu_manager);

    [[maybe_unused]] int nb_enabled_cams = camera_manager.EnableCameras();
    std::thread camera_thread = std::thread(&CameraManager::RunLoop, &camera_manager);

    // Helper to stop threads before any early return below this point.
    auto stop_threads = [&]() {
        imu_manager.StopLoop();
        if (imu_thread.joinable()) imu_thread.join();
        camera_manager.StopLoops();
        if (camera_thread.joinable()) camera_thread.join();
    };

    try
    {
        auto ds = DatasetManager::GetActiveDatasetManager(DATASET_KEY_CMD);

        if (ds) // if already exists
        {
            // need to ensure it's actually running
            if (ds->Running())
            {
                // if running: TODO: return ERROR ACK saying that a dataset is already running
                // If completed, stop it then too
                SPDLOG_ERROR("Dataset already running under key {}, ignoring command", DATASET_KEY_CMD);
                stop_threads();
                return to_uint8(EC::PLACEHOLDER);
            }
            else
            {
                ds->StopDatasetManager(DATASET_KEY_CMD); // remove it (will create a new one)
            }
        }

        // Create a new Dataset
        SPDLOG_INFO("Starting dataset collection (type {}) for {} frames at a period of {} seconds.",
                    static_cast<uint8_t>(ds_config.capture_mode), ds_config.target_frame_nb, ds_config.maximum_period);

        ds = DatasetManager::Create(ds_config, DATASET_KEY_CMD, camera_manager, imu_manager, inference_manager);
        ds->StartCollection();

        // StartCollection is asynchronous — block here until the collection loop
        // finishes naturally (frame target met or period elapsed).
        while (ds->Running())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        const std::string dataset_folder = ds->GetDatasetFolder();
        ds->StopDatasetManager(DATASET_KEY_CMD);
        WriteResult(out_path, dataset_folder);
    }
    catch (...)
    {
        stop_threads();
        throw;
    }

    stop_threads();

    return 0;
}

int main(int argc, char** argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception& e)
    {
        spdlog::critical("Unhandled exception: {}", e.what());
        return to_uint8(EC::PLACEHOLDER);
    }
    catch (...)
    {
        spdlog::critical("Unhandled unknown exception");
        return to_uint8(EC::PLACEHOLDER);
    }
}
