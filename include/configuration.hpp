#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "core/toml.hpp"

#define NUM_CAMERAS 4


struct CameraConfig {
    bool enable;
    int64_t id;
    std::string path;
    int64_t width;
    int64_t height;
};



class Configuration
{
public:
    Configuration();
    void LoadConfiguration(std::string config_path);
    const std::array<CameraConfig, NUM_CAMERAS>& GetCameraConfigs() const;
    bool UpdateCameraConfigs(const std::array<CameraConfig, NUM_CAMERAS>& new_configs);

private:
    std::string config_path;
    bool configured;
    toml::table config;
    toml::table* camera_devices_config;
    std::array<CameraConfig, NUM_CAMERAS> camera_configs;
    void ParseCameraDevicesConfig();
};;





#endif // CONFIGURATION_HPP