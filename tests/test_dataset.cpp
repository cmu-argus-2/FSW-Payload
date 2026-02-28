/*
    Test file to replicate what should happen in FSW when the command to start 
    dataset collection is received.
*/
#include "spdlog/spdlog.h"
#include "vision/dataset.hpp"
#include "core/timing.hpp"
#include "configuration.hpp"
#include <memory>
#include <thread>
#include <array>

#define DATASET_KEY_CMD "CMD"


int main(int argc, char** argv)
{
    // Dataset collection configuration flag
    auto config = std::make_unique<Configuration>();
    config->LoadConfiguration("config/config.toml");

    CAPTURE_MODE capture_mode = CAPTURE_MODE::PERIODIC; // default to periodic
    double period = 60.0; // default to 60s
    uint16_t nb_frames = 4;

    // collect IMU data or not flag
    const auto& imu_config = config->GetIMUConfig();
    IMUManager imu_manager(imu_config);

    const auto& cam_configs = config->GetCameraConfigs();
    CameraManager camera_manager(cam_configs);

    // TODO: Create the IMUManager and CameraManager instances and threads for the 
    // datasetManager to interface with, preferably without having to create 
    // a payload instance for this test
    std::thread imu_thread = std::thread(&IMUManager::RunLoop, &imu_manager);

    std::array<bool, NUM_CAMERAS> temp;
    [[maybe_unused]] int nb_enabled_cams = camera_manager.EnableCameras(temp);
    std::thread camera_thread = std::thread(&CameraManager::RunLoop, &camera_manager);

    // how to make the above accessible to the dataset manager through the sys namespace?

    if (period == 0.0 || nb_frames == 0)
    {
        spdlog::error("Invalid parameters for dataset collection command");
        return 0;
    }

    auto ds = DatasetManager::GetActiveDataset(DATASET_KEY_CMD);

    if (ds) // if already exists
    {
        // need to ensure it's actually running
        if (ds->Running())
        {
            // if running: TODO: return ERROR ACK saying that a dataset is already running
            // If completed, stop it then too
        }
        else
        {
            ds->StopDataset(DATASET_KEY_CMD); // remove it (will create a new one)
        }
    }

    // Create a new Dataset
    SPDLOG_INFO("Starting dataset collection (type {}) for {} frames at a period of {} seconds.", static_cast<uint8_t>(capture_mode), nb_frames, period);

    ds = DatasetManager::Create(period, nb_frames, capture_mode, timing::GetCurrentTimeMs(),
                                IMU_COLLECTION_MODE::GYRO_MAG_TEMP, uint8_t(30), 1.0f, ProcessingStage::NotPrefiltered,
                                DATASET_KEY_CMD, camera_manager, imu_manager);
    ds->StartCollection();

    // close threads
    imu_manager.StopLoop();
    if (imu_thread.joinable()) imu_thread.join();

    camera_manager.StopLoops();
    if (camera_thread.joinable()) camera_thread.join();

    return 0;
}