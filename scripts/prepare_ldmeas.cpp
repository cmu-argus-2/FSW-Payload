/*
  prepare_ldmeas <dataset_folder> [config_path]

  Converts a completed dataset into a landmark_measurements.csv file
  that can be fed directly to the batch optimizer.

  The CSV is written to <dataset_folder>/landmark_measurements.csv and
  contains one row per landmark observation:
    timestamp_ms, bearing_x, bearing_y, bearing_z,
    eci_x_km, eci_y_km, eci_z_km, group, sigma

  Arguments:
    dataset_folder   — path to a dataset folder with LDNeted frame JSONs
                       and an imu_data.csv (e.g. data/datasets/1234567890/)
    config_path      — system configuration TOML with camera calibration
                       (default: config/config.toml)
*/

#include <string>
#include <iostream>

#include "spdlog/spdlog.h"
#include "configuration.hpp"
#include "navigation/od.hpp"

static constexpr const char* kDefaultConfigPath    = "config/config.toml";

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: prepare_ldmeas <dataset_folder> [config_path]\n";
        return 1;
    }
    spdlog::set_level(spdlog::level::info);

    const std::string dataset_folder  = argv[1];
    const std::string config_path     = (argc > 2) ? argv[2] : kDefaultConfigPath;

    spdlog::info("prepare_ldmeas: dataset={} config={}", dataset_folder, config_path);

    Configuration config;
    try
    {
        config.LoadConfiguration(config_path);
    }
    catch (const std::exception& e)
    {
        spdlog::error("Failed to load configuration from {}: {}", config_path, e.what());
        return 1;
    }

    const CameraCalibration& calibration = config.GetCameraCalibration();

    OD od;
    try
    {
        const ErrorCode code = od.DatasetPrepare(dataset_folder, calibration);
        if (code != ErrorCode::OK)
        {
            spdlog::error("prepare_ldmeas: DatasetPrepare failed for {} with error code {}",
                          dataset_folder, static_cast<int>(code));
            return 1;
        }
    }
    catch (const std::exception& e)
    {
        spdlog::error("prepare_ldmeas: DatasetPrepare threw an exception: {}", e.what());
        return 1;
    }
    catch (...)
    {
        spdlog::error("prepare_ldmeas: DatasetPrepare threw an unknown exception");
        return 1;
    }

    spdlog::info("prepare_ldmeas: landmark_measurements.csv written to {}/",
                 dataset_folder);
    return 0;
}
