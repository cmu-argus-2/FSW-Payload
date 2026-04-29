/*
  run_od_pipeline <dataset_config_folder> [od_config_path] [system_config_path] [--out <out_path>]

  Captures a dataset from the dataset config, processes it to the requested
  stage, prepares OD measurements if needed, runs batch OD, and writes results.

    --out  File to write the generated results directory path into. Falls back
           to path.out if not provided or not writable.
*/

#include "configuration.hpp"
#include "core/timing.hpp"
#include "inference/inference_manager.hpp"
#include "navigation/od.hpp"
#include "vision/dataset.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>
#include "toml.hpp"

static constexpr const char* kDefaultODConfigPath     = "config/od.toml";
static constexpr const char* kDefaultSystemConfigPath = "config/config.toml";
static constexpr const char* kDefaultOutPath          = "path.out";

// Returns the value of --flag, or default_val if not present.
static std::string GetFlag(int argc, char** argv, const char* flag, const char* default_val = "")
{
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == flag) return argv[i + 1];
    return default_val;
}

// Returns positional args, skipping any --flag <value> pairs.
static std::vector<std::string> GetPositional(int argc, char** argv)
{
    std::vector<std::string> result;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.size() > 2 && a[0] == '-' && a[1] == '-') { ++i; continue; }
        result.push_back(a);
    }
    return result;
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

static bool LoadDatasetConfig(const std::string& folder_path, DatasetConfig& out)
{
    const std::string path = folder_path + "/dataset_config.toml";
    try {
        const toml::table cfg = toml::parse_file(path);
        out.maximum_period = cfg["maximum_period"].value_or(out.maximum_period);
        const uint64_t target_frame_nb =
            cfg["target_frame_nb"].value_or(uint64_t(out.target_frame_nb));
        if (target_frame_nb == 0 || target_frame_nb > MAX_SAMPLES) {
            spdlog::error("Invalid target_frame_nb {} in {}", target_frame_nb, path);
            return false;
        }
        out.target_frame_nb = static_cast<uint8_t>(target_frame_nb);
        out.capture_mode = static_cast<CAPTURE_MODE>(
            cfg["dataset_capture_mode"].value_or(uint64_t(out.capture_mode)));
        out.imu_collection_mode = static_cast<IMU_COLLECTION_MODE>(
            cfg["imu_collection_mode"].value_or(uint64_t(out.imu_collection_mode)));
        out.image_capture_rate = static_cast<uint8_t>(
            cfg["image_capture_rate"].value_or(uint64_t(out.image_capture_rate)));
        out.imu_sample_rate_hz = static_cast<float>(
            cfg["imu_sample_rate_hz"].value_or(double(out.imu_sample_rate_hz)));
        out.target_processing_stage = static_cast<ProcessingStage>(
            cfg["target_processing_stage"].value_or(uint64_t(out.target_processing_stage)));
        if (const auto* arr = cfg["active_cameras"].as_array())
            for (size_t i = 0; i < NUM_CAMERAS && i < arr->size(); ++i)
                if (auto val = (*arr)[i].value<bool>()) out.active_cameras[i] = *val;
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse dataset config {}: {}", path, e.what());
        return false;
    }

    if (out.capture_start_time == 0) {
        out.capture_start_time = timing::GetCurrentTimeMs();
    }
    return Dataset::isValidConfiguration(out.maximum_period,
                                         out.target_frame_nb,
                                         out.capture_mode,
                                         out.imu_collection_mode,
                                         out.image_capture_rate,
                                         out.imu_sample_rate_hz,
                                         out.target_processing_stage,
                                         out.capture_start_time);
}

int main(int argc, char** argv)
{
    const auto positional = GetPositional(argc, argv);
    if (positional.empty()) {
        std::cerr << "Usage: run_od_pipeline <dataset_config_folder> "
                     "[od_config_path] [system_config_path] [--out <out_path>]\n";
        return 1;
    }
    spdlog::set_level(spdlog::level::info);

    const std::string out_path = ResolveOutPath(
        GetFlag(argc, argv, "--out", kDefaultOutPath));

    ODRequest request;
    if (!LoadDatasetConfig(positional[0], request.dataset_config)) {
        return 1;
    }
    request.od_config_path     = positional.size() > 1 ? positional[1] : kDefaultODConfigPath;
    request.system_config_path = positional.size() > 2 ? positional[2] : kDefaultSystemConfigPath;

    auto config = std::make_unique<Configuration>();
    try {
        config->LoadConfiguration(request.system_config_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load system configuration {}: {}",
                      request.system_config_path, e.what());
        return 1;
    }

    InferenceManager inference_manager;
    IMUManager imu_manager(config->GetIMUConfig());
    CameraManager camera_manager(config->GetCameraConfigs(),
                                 config->GetCameraISPConfig(),
                                 inference_manager);

    std::thread imu_thread(&IMUManager::RunLoop, &imu_manager);
    [[maybe_unused]] int nb_enabled_cams = camera_manager.EnableCameras();
    std::thread camera_thread(&CameraManager::RunLoop, &camera_manager);

    auto stop_threads = [&]() {
        imu_manager.StopLoop();
        if (imu_thread.joinable()) imu_thread.join();
        camera_manager.StopLoops();
        if (camera_thread.joinable()) camera_thread.join();
    };

    const ODResult result = RunODPipeline(request, camera_manager, imu_manager, inference_manager);
    stop_threads();

    if (result.code != ErrorCode::OK) {
        spdlog::error("OD pipeline failed at stage {} with error code {}.",
                      static_cast<int>(result.stage), static_cast<int>(result.code));
        return 1;
    }

    spdlog::info("OD complete. Dataset {} results in {}",
                 result.dataset_folder, result.results_dir);
    WriteResult(out_path, result.results_dir);
    return 0;
}
