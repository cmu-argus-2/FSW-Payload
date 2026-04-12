#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include "toml.hpp"
#include "vision/camera_manager.hpp"
#include "imu/imu_manager.hpp"
#include <opencv2/core.hpp>

#define NUM_CAMERAS 4

// Optical calibration shared across all cameras (same lens).
// cam_to_body[i] is the rotation matrix from camera i's frame to the
// spacecraft body frame.
struct CameraCalibration
{
    cv::Mat camera_matrix;                      // 3×3 CV_64F intrinsic matrix K
    cv::Mat dist_coeffs;                        // 1×5 CV_64F [k1, k2, p1, p2, k3]
    std::array<cv::Mat, NUM_CAMERAS> cam_to_body; // per-camera 3×3 CV_64F rotation
};

class Configuration
{
public:
    Configuration();
    void LoadConfiguration(std::string config_path);
    const std::array<CameraConfig, NUM_CAMERAS>& GetCameraConfigs() const;
    const IMUConfig& GetIMUConfig() const;
    const CameraCalibration& GetCameraCalibration() const;

private:
    bool configured;
    std::string config_path;
    toml::table config;
    toml::table* camera_devices_config;
    toml::table* imu_config_table;
    std::array<CameraConfig, NUM_CAMERAS> camera_configs;
    IMUConfig imu_config;
    CameraCalibration camera_calibration;
    void ParseCameraDevicesConfig();
    void ParseIMUConfig();
    void ParseCameraCalibration();
};


template <typename T>
T get_or_warn(const toml::table& t, std::string_view key, T def, std::string_view msg_key);


#endif // CONFIGURATION_HPP