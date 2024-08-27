#include "configuration.hpp"

Configuration::Configuration()
: config()
{
}


void Configuration::LoadConfiguration(std::string config_path)
{
    this->config = toml::parse_file(config_path);
    this->camera_devices_config = *config["camera-devices"].as_table();


    // Parse camera configurations after loading the config
    ParseCameraDevicesConfig();

}

void Configuration::ParseCameraDevicesConfig()
{
    // Iterate over each camera entry in camera-devices
    for (const auto& [key, value] : camera_devices_config) {
        auto cam_table = value.as_table();
        if (cam_table) {
            CameraConfig cam_config;
            cam_config.enable = cam_table->get_as<bool>("enable")->value_or(false);

            cam_config.path = cam_table->get_as<std::string>("path")->value_or("");

            // If the path is empty, disable the camera
            if (cam_config.path.empty()) {
                cam_config.enable = false;
            }

            if (const auto* resolution = cam_table->get("resolution")->as_table()) {
                cam_config.width = resolution->get_as<int64_t>("width")->value_or(640);
                cam_config.height = resolution->get_as<int64_t>("height")->value_or(480);
            } else {
                cam_config.width = 640;
                cam_config.height = 480;
            }

            camera_configs.push_back(cam_config);
        }
    }
}