#include "configuration.hpp"
#include "spdlog/spdlog.h"
#include <string>
#include "iostream"

Configuration::Configuration()
: configured(false), config(), camera_devices_config(nullptr)
{
}


void Configuration::LoadConfiguration(std::string config_path)
{
    this->config_path = config_path;
    this->config = toml::parse_file(config_path);
    this->camera_devices_config = config["camera-device"].as_table();


    // Parse camera configurations after loading the config
    ParseCameraDevicesConfig();


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

            cam_config.enable = cam_table->get_as<bool>("enable")->value_or(false);
            cam_config.path = cam_table->get_as<std::string>("path")->value_or("");

            // If the path is empty, disable the camera
            if (cam_config.path.empty()) {
                cam_config.enable = false;
            }

            cam_config.id = cam_table->get_as<int64_t>("id")->value_or(0);
            cam_config.width = cam_table->get_as<int64_t>("resolution_width")->value_or(640);
            cam_config.height = cam_table->get_as<int64_t>("resolution_height")->value_or(480);


            camera_configs[idx] = cam_config;
            idx++;
        }
    }
}

const std::array<CameraConfig, NUM_CAMERAS>& Configuration::GetCameraConfigs() const
{
    return camera_configs;
}



bool Configuration::UpdateCameraConfigs(const std::array<CameraConfig, NUM_CAMERAS>& new_configs)
{
    if (!configured) {
        SPDLOG_ERROR("Configuration not loaded");
        return false;
    }
    

    // Update the camera devices config
    for (auto new_config : new_configs) 
    {
        (*config["camera-device"]["cam" + std::to_string(new_config.id)].as_table()).insert_or_assign("enable", new_config.enable);
    }
    
    // Re-Serialize toml file
    std::ofstream out(config_path);
    out << config;
    out.close();

    SPDLOG_INFO("Configuration updated successfully");


    return true;

}
