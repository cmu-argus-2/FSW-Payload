/*
  run_od_pipeline <dataset_config_folder> [od_config_path] [system_config_path]

  Captures a dataset from the dataset config, processes it to the requested
  stage, prepares OD measurements if needed, runs batch OD, and writes results.
*/

#include "configuration.hpp"
#include "core/timing.hpp"
#include "inference/inference_manager.hpp"
#include "navigation/od.hpp"
#include "vision/dataset.hpp"

#include <iostream>
#include <memory>
#include <thread>

#include <spdlog/spdlog.h>
#include "toml.hpp"

static constexpr const char* kDefaultODConfigPath = "config/od.toml";
static constexpr const char* kDefaultSystemConfigPath = "config/config.toml";

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
    if (argc < 2) {
        std::cerr << "Usage: run_od_pipeline <dataset_config_folder> "
                     "[od_config_path] [system_config_path]\n";
        return 1;
    }
    spdlog::set_level(spdlog::level::info);

    ODRequest request;
    if (!LoadDatasetConfig(argv[1], request.dataset_config)) {
        return 1;
    }
    request.od_config_path = (argc > 2) ? argv[2] : kDefaultODConfigPath;
    request.system_config_path = (argc > 3) ? argv[3] : kDefaultSystemConfigPath;

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
    return 0;
}
