#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "toml.hpp"
#include "vision/camera_manager.hpp"
#include "imu/imu_manager.hpp"

#define NUM_CAMERAS 4


class Configuration
{
public:
    Configuration();
    void LoadConfiguration(std::string config_path);
    const std::array<CameraConfig, NUM_CAMERAS>& GetCameraConfigs() const;
    const IMUConfig& GetIMUConfig() const;

private:
    bool configured;
    std::string config_path;
    toml::table config;
    toml::table* camera_devices_config;
    toml::table* imu_config_table;
    std::array<CameraConfig, NUM_CAMERAS> camera_configs;
    IMUConfig imu_config;
    void ParseCameraDevicesConfig();
    void ParseIMUConfig();
};


template <typename T>
T get_or_warn(const toml::table& t, std::string_view key, T def, std::string_view msg_key);


#endif // CONFIGURATION_HPP