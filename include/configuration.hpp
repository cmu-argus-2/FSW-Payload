#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "toml.hpp"


struct CameraConfig {
    bool enable;
    std::string path;
    int64_t width;
    int64_t height;
};



class Configuration
{

public:


    Configuration();

    void LoadConfiguration(std::string config_path);
    std::vector<CameraConfig> GetCameraConfigs() const;

private:

    toml::table config;
    toml::table camera_devices_config;
    std::vector<CameraConfig> camera_configs;
    
    void ParseCameraDevicesConfig();


};





#endif // CONFIGURATION_HPP