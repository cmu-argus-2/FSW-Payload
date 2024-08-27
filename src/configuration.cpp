#include "configuration.hpp"
#include "spdlog/spdlog.h"

Configuration::Configuration()
: config()
{
}


void Configuration::LoadConfiguration(std::string config_path)
{
    this->config = toml::parse_file(config_path);
    this->camera_devices_config = *config["camera-device"].as_table();


    // Parse camera configurations after loading the config
    ParseCameraDevicesConfig();

}

void Configuration::ParseCameraDevicesConfig()
{
    // Check if the number of camera devices is exactly 4
    if (camera_devices_config.size() != 4) {
        throw std::runtime_error("Invalid number of camera devices. Expected 4.");
        SPDLOG_CRITICAL("Invalid number of camera devices. Expected 4.");
    }
    
    // Iterate over each camera entry in camera-devices
    int idx = 0;
    for (const auto& [key, value] : camera_devices_config) 
    {
        auto cam_table = value.as_table();
        if (cam_table) {
            CameraConfig cam_config;
            cam_config.enable = cam_table->get_as<bool>("enable")->value_or(false);

            cam_config.path = cam_table->get_as<std::string>("path")->value_or("");

            // If the path is empty, disable the camera
            if (cam_config.path.empty()) {
                cam_config.enable = false;
            }

            cam_config.id = cam_table->get_as<int64_t>("id")->value_or(0);

            if (const auto* resolution = cam_table->get("resolution")->as_table()) {
                cam_config.width = resolution->get_as<int64_t>("width")->value_or(640);
                cam_config.height = resolution->get_as<int64_t>("height")->value_or(480);
            } else {
                cam_config.width = 640;
                cam_config.height = 480;
            }

            camera_configs[idx] = cam_config;
            idx++;
        }
    }
}


const std::array<CameraConfig, NUM_CAMERAS>& Configuration::GetCameraConfigs() const
{
    return camera_configs;
}