#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "toml.hpp"
#include "vision/camera_manager.hpp"

#define NUM_CAMERAS 4


class Configuration
{
public:
    Configuration();
    void LoadConfiguration(std::string config_path);
    const std::array<CameraConfig, NUM_CAMERAS>& GetCameraConfigs() const;

private:
    bool configured;
    std::string config_path;
    toml::table config;
    toml::table* camera_devices_config;
    std::array<CameraConfig, NUM_CAMERAS> camera_configs;
    void ParseCameraDevicesConfig();
};





#endif // CONFIGURATION_HPP