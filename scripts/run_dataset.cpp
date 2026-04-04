/*
    Test file to replicate what should happen in FSW when the command to start 
    dataset collection is received.
*/
#include "spdlog/spdlog.h"
#include "vision/dataset_manager.hpp"
#include "inference/inference_manager.hpp"
#include "core/timing.hpp"
#include "configuration.hpp"
#include <memory>
#include <thread>
#include <array>

#define DATASET_KEY_CMD "CMD"


int main(int argc, char** argv)
{
    std::string config_file_path = "config/config.toml";
    std::string ds_config_folder_path = "config";
    // Dataset collection configuration flag
    auto config = std::make_unique<Configuration>();
    try {
        config->LoadConfiguration(config_file_path);
    } catch (const toml::parse_error& err) {
        std::cerr << "Parsing configuration file failed: " << err << "\n";
        return 1;
    }
    SPDLOG_INFO("Configuration file {} loaded.", config_file_path);

    CAPTURE_MODE capture_mode = CAPTURE_MODE::PERIODIC;
    IMU_COLLECTION_MODE imu_collection_mode = IMU_COLLECTION_MODE::GYRO_MAG_TEMP;
    double max_period = 10.0;
    uint16_t target_frame_nb = 4;
    uint8_t image_capture_rate = uint8_t(1);
    float imu_sample_rate_hz = 1.0f;
    ProcessingStage target_processing_stage = ProcessingStage::NotPrefiltered;

    // Load dataset parameters from config file
    const std::string ds_config_path = ds_config_folder_path + "/dataset_config.toml";
    try {
        toml::table ds_cfg = toml::parse_file(ds_config_path);
        max_period              = ds_cfg["maximum_period"].value_or(max_period);
        target_frame_nb         = static_cast<uint16_t>(ds_cfg["target_frame_nb"].value_or(uint64_t(target_frame_nb)));
        capture_mode            = static_cast<CAPTURE_MODE>(ds_cfg["dataset_capture_mode"].value_or(uint64_t(capture_mode)));
        imu_collection_mode     = static_cast<IMU_COLLECTION_MODE>(ds_cfg["imu_collection_mode"].value_or(uint64_t(imu_collection_mode)));
        image_capture_rate      = static_cast<uint8_t>(ds_cfg["image_capture_rate"].value_or(uint64_t(image_capture_rate)));
        imu_sample_rate_hz      = static_cast<float>(ds_cfg["imu_sample_rate_hz"].value_or(double(imu_sample_rate_hz)));
        target_processing_stage = static_cast<ProcessingStage>(ds_cfg["target_processing_stage"].value_or(uint64_t(target_processing_stage)));
    } catch (const toml::parse_error& err) {
        spdlog::error("Failed to parse dataset config {}: {}", ds_config_path, err.description());
        return 1;
    }
    // capture_start_time = 0 in config means "start immediately"
    int64_t capture_start_time = timing::GetCurrentTimeMs();

    // collect IMU data or not flag
    const auto& imu_config = config->GetIMUConfig();
    IMUManager imu_manager(imu_config);

    InferenceManager inference_manager;

    const auto& cam_configs = config->GetCameraConfigs();
    CameraManager camera_manager(cam_configs, inference_manager);

    std::thread imu_thread = std::thread(&IMUManager::RunLoop, &imu_manager);

    std::array<bool, NUM_CAMERAS> temp;
    [[maybe_unused]] int nb_enabled_cams = camera_manager.EnableCameras(temp);
    std::thread camera_thread = std::thread(&CameraManager::RunLoop, &camera_manager);

    // how to make the above accessible to the dataset manager through the sys namespace?

    if (max_period == 0.0 || target_frame_nb == 0)
    {
        spdlog::error("Invalid parameters for dataset collection command");
        return 0;
    }

    auto ds = DatasetManager::GetActiveDatasetManager(DATASET_KEY_CMD);

    if (ds) // if already exists
    {
        // need to ensure it's actually running
        if (ds->Running())
        {
            // if running: TODO: return ERROR ACK saying that a dataset is already running
            // If completed, stop it then too
            SPDLOG_ERROR("Dataset already running under key {}, ignoring command", DATASET_KEY_CMD);
            imu_manager.StopLoop();
            if (imu_thread.joinable()) imu_thread.join();
            camera_manager.StopLoops();
            if (camera_thread.joinable()) camera_thread.join();
            return 1;
        }
        else
        {
            ds->StopDatasetManager(DATASET_KEY_CMD); // remove it (will create a new one)
        }
    }

    // Create a new Dataset
    SPDLOG_INFO("Starting dataset collection (type {}) for {} frames at a period of {} seconds.", static_cast<uint8_t>(capture_mode), target_frame_nb, max_period);

    ds = DatasetManager::Create(max_period, target_frame_nb, capture_mode, capture_start_time,
                                imu_collection_mode, image_capture_rate, imu_sample_rate_hz,
                                target_processing_stage, DATASET_KEY_CMD, camera_manager, imu_manager, inference_manager);
    ds->StartCollection();

    // StartCollection is asynchronous — block here until the collection loop
    // finishes naturally (frame target met or period elapsed).
    while (ds->Running())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Close Dataset Manager (joins the collection thread internally)
    ds->StopDatasetManager(DATASET_KEY_CMD);

    // close threads
    imu_manager.StopLoop();
    if (imu_thread.joinable()) imu_thread.join();

    camera_manager.StopLoops();
    if (camera_thread.joinable()) camera_thread.join();

    return 0;
}
