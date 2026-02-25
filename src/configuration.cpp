#include "configuration.hpp"
#include <string>
#include <iostream>
#include "spdlog/spdlog.h"

Configuration::Configuration()
: configured(false), config(), camera_devices_config(nullptr), imu_config_table(nullptr)
{
}


void Configuration::LoadConfiguration(std::string config_path)
{
    this->config_path = config_path;
    this->config = toml::parse_file(config_path);
    this->camera_devices_config = config["camera-device"].as_table();
    this->imu_config_table = config["imu-device"].as_table();

    // Parse camera configurations after loading the config
    ParseCameraDevicesConfig();

    // Parse IMU configuration
    ParseIMUConfig();

    this->configured = true;

}

void Configuration::ParseCameraDevicesConfig()
{
    // Check if the number of camera devices is exactly 4
    if (camera_devices_config->size() != 4) {
        throw std::runtime_error("Invalid number of camera devices. Expected 4.");
        SPDLOG_CRITICAL("Invalid number of camera devices. Expected 4.");
    }
    
    
    // Iterate over each camera entry in camera-devices
    std::size_t idx = 0;
    for (const auto& [key, value] : *camera_devices_config) 
    {
        auto cam_table = value.as_table();

        if (cam_table) {
            CameraConfig cam_config;
            // TODO (Error handling): some of these shouldn't be optional
            cam_config.id     = get_or_warn<int64_t>(*cam_table, "id", 0, "id");
            cam_config.width  = get_or_warn<int64_t>(*cam_table, "resolution_width", 640, "resolution_width");
            cam_config.height = get_or_warn<int64_t>(*cam_table, "resolution_height", 480, "resolution_height");

            if (auto p = cam_table->get_as<std::string>("path"))
                cam_config.path = (*p).get();
            else {
                SPDLOG_WARN("Failed to parse 'path' from camera config. Using default value: empty string");
                cam_config.path = "";
            }

            camera_configs[idx] = cam_config;
            ++idx;
        }
    }
}

void Configuration::ParseIMUConfig()
{
    if (!imu_config_table) {
        throw std::runtime_error("IMU configuration section missing in config file.");
        SPDLOG_CRITICAL("IMU configuration section missing in config file.");
    }

    auto chipid = imu_config_table->get_as<int64_t>("chipid");
    if (!chipid) {
        SPDLOG_WARN("Failed to parse 'chipid' from IMU config. Using default value: 0xD8");
        imu_config.chipid = 0xD8;
    } else {
        imu_config.chipid = static_cast<uint8_t>((*chipid).get());
    }

    auto i2c_addr = imu_config_table->get_as<int64_t>("i2c_addr");
    if (!i2c_addr) {
        SPDLOG_WARN("Failed to parse 'i2c_addr' from IMU config. Using default value: 0x68");
        imu_config.i2c_addr = 0x68;
    } else {
        imu_config.i2c_addr = static_cast<uint8_t>((*i2c_addr).get());
    }

    auto i2c_path = imu_config_table->get_as<std::string>("i2c_path");
    if (!i2c_path) {
        SPDLOG_WARN("Failed to parse 'i2c_path' from IMU config. Using default value: /dev/i2c-7");
        imu_config.i2c_path = "/dev/i2c-7";
    } else {
        imu_config.i2c_path = (*i2c_path).get();
    }
}

const std::array<CameraConfig, NUM_CAMERAS>& Configuration::GetCameraConfigs() const
{
    return camera_configs;
}


const IMUConfig& Configuration::GetIMUConfig() const
{
    return imu_config;
}

template <typename T> T get_or_warn(const toml::table& t, std::string_view key, T def, std::string_view msg_key)
{
    if (auto v = t.get_as<T>(key))
        return (*v).get();
    SPDLOG_WARN("Failed to parse '{}' from camera config. Using default value: {}", msg_key, def);
    return def;
}