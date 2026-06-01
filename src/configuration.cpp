#include "configuration.hpp"
#include "inference/inference_manager.hpp"
#include "inference/types.hpp"
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
    this->camera_isp_config = CameraISPConfig{};

    // Parse camera configurations after loading the config
    ParseCameraDevicesConfig();
    ParseCameraISPConfig();

    // Parse IMU configuration
    ParseIMUConfig();

    // Parse camera calibration
    ParseCameraCalibration();

    this->configured = true;

}

void Configuration::ParseCameraDevicesConfig()
{
    if (!camera_devices_config) {
        SPDLOG_CRITICAL("Camera configuration section missing in config file.");
        throw std::runtime_error("Camera configuration section missing in config file.");
    }

    // Check if the number of camera devices is exactly 4
    if (camera_devices_config->size() != 4) {
        SPDLOG_CRITICAL("Invalid number of camera devices. Expected 4.");
        throw std::runtime_error("Invalid number of camera devices. Expected 4.");
    }

    for (auto& cfg : camera_configs)
    {
        cfg = CameraConfig{};
    }

    std::array<bool, NUM_CAMERAS> seen{};

    // Iterate over each camera entry in camera-devices
    for (const auto& [key, value] : *camera_devices_config) 
    {
        auto cam_table = value.as_table();

        if (cam_table) {
            CameraConfig cam_config;
            // TODO (Error handling): some of these shouldn't be optional
            cam_config.id      = get_or_warn<int64_t>(*cam_table, "id", 0, "id");
            cam_config.width   = get_or_warn<int64_t>(*cam_table, "resolution_width", 640, "resolution_width");
            cam_config.height  = get_or_warn<int64_t>(*cam_table, "resolution_height", 480, "resolution_height");
            cam_config.enabled = get_or_warn<bool>(*cam_table, "enabled", true, "enabled");

            if (auto p = cam_table->get_as<std::string>("path"))
                cam_config.path = (*p).get();
            else {
                SPDLOG_WARN("Failed to parse 'path' from camera config. Using default value: empty string");
                cam_config.path = "";
            }

            if (cam_config.id < 0 || cam_config.id >= NUM_CAMERAS)
            {
                SPDLOG_CRITICAL("Camera '{}' has invalid id {}. Expected range [0, {}].",
                                key.str(), cam_config.id, NUM_CAMERAS - 1);
                throw std::runtime_error("Camera configuration has invalid id.");
            }

            const std::size_t idx = static_cast<std::size_t>(cam_config.id);
            if (seen[idx])
            {
                SPDLOG_CRITICAL("Duplicate camera id {} in camera configuration.", cam_config.id);
                throw std::runtime_error("Duplicate camera id in configuration.");
            }

            camera_configs[idx] = cam_config;
            seen[idx] = true;
        }
    }
}

void Configuration::ParseCameraISPConfig()
{
    camera_isp_config = CameraISPConfig{};

    auto* tbl = config["camera-isp"].as_table();
    if (!tbl) {
        SPDLOG_INFO("No [camera-isp] section found in config — using all defaults.");
        return;
    }

    // Only assign fields that are explicitly present in the TOML.
    // Absent fields keep the struct defaults declared in CameraISPConfig.
    if (auto v = tbl->get_as<int64_t>("wbmode"))               camera_isp_config.wbmode               = static_cast<int>((*v).get());
    if (auto v = tbl->get_as<bool>("aelock"))                   camera_isp_config.aelock               = (*v).get();
    if (auto v = tbl->get_as<bool>("awblock"))                  camera_isp_config.awblock              = (*v).get();
    if (auto v = tbl->get_as<int64_t>("ee_mode"))               camera_isp_config.ee_mode              = static_cast<int>((*v).get());
    if (auto v = tbl->get_as<double>("ee_strength"))            camera_isp_config.ee_strength          = static_cast<float>((*v).get());
    if (auto v = tbl->get_as<int64_t>("aeantibanding"))         camera_isp_config.aeantibanding        = static_cast<int>((*v).get());
    if (auto v = tbl->get_as<double>("exposurecompensation"))   camera_isp_config.exposurecompensation = static_cast<float>((*v).get());
    if (auto v = tbl->get_as<int64_t>("tnr_mode"))              camera_isp_config.tnr_mode             = static_cast<int>((*v).get());
    if (auto v = tbl->get_as<double>("tnr_strength"))           camera_isp_config.tnr_strength         = static_cast<float>((*v).get());
    if (auto v = tbl->get_as<double>("saturation"))             camera_isp_config.saturation           = static_cast<float>((*v).get());
    if (auto v = tbl->get_as<int64_t>("fps"))                   camera_isp_config.fps                  = static_cast<int>((*v).get());
    if (auto v = tbl->get_as<int64_t>("max_buffers"))           camera_isp_config.max_buffers          = static_cast<int>((*v).get());

    if (auto arr = tbl->get_as<toml::array>("exposuretimerange"); arr && arr->size() == 2) {
        auto lo = arr->get_as<int64_t>(0);
        auto hi = arr->get_as<int64_t>(1);
        if (lo && hi)
            camera_isp_config.exposuretimerange = {(*lo).get(), (*hi).get()};
    }
    if (auto arr = tbl->get_as<toml::array>("gainrange"); arr && arr->size() == 2) {
        auto lo = arr->get_as<double>(0);
        auto hi = arr->get_as<double>(1);
        if (lo && hi)
            camera_isp_config.gainrange = {static_cast<float>((*lo).get()), static_cast<float>((*hi).get())};
    }
    if (auto arr = tbl->get_as<toml::array>("ispdigitalgainrange"); arr && arr->size() == 2) {
        auto lo = arr->get_as<double>(0);
        auto hi = arr->get_as<double>(1);
        if (lo && hi)
            camera_isp_config.ispdigitalgainrange = {static_cast<float>((*lo).get()), static_cast<float>((*hi).get())};
    }
}

void Configuration::ParseIMUConfig()
{
    if (!imu_config_table) {
        SPDLOG_CRITICAL("IMU configuration section missing in config file.");
        throw std::runtime_error("IMU configuration section missing in config file.");
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

const CameraISPConfig& Configuration::GetCameraISPConfig() const
{
    return camera_isp_config;
}


const IMUConfig& Configuration::GetIMUConfig() const
{
    return imu_config;
}

const CameraCalibration& Configuration::GetCameraCalibration() const
{
    return camera_calibration;
}

EC Configuration::ApplyInferenceConfig(InferenceManager& inference_manager) const
{
    const auto* inference_table = config["inference"].as_table();
    if (!inference_table) {
        SPDLOG_INFO("No [inference] section found in config — using InferenceManager defaults.");
        return EC::OK;
    }

    if (auto v = inference_table->get_as<int64_t>("rc_version")) {
        const int version = static_cast<int>((*v).get());
        if (version <= 0) {
            SPDLOG_ERROR("Invalid inference.rc_version {}; expected > 0.", version);
            return EC::INVALID_COMMAND_ARGUMENTS;
        }
        EC ec = inference_manager.SetRCNetVersion(version);
        if (ec != EC::OK) return ec;
    }

    if (auto v = inference_table->get_as<int64_t>("ld_version")) {
        const int version = static_cast<int>((*v).get());
        if (version <= 0) {
            SPDLOG_ERROR("Invalid inference.ld_version {}; expected > 0.", version);
            return EC::INVALID_COMMAND_ARGUMENTS;
        }
        EC ec = inference_manager.SetLDNetVersion(version);
        if (ec != EC::OK) return ec;
    }

    const auto* ldnet_table = (*inference_table)["ldnet"].as_table();
    if (!ldnet_table) return EC::OK;

    LDNetConfig ldnet_config = inference_manager.GetLDNetConfig();
    bool ldnet_changed = false;

    if (auto v = ldnet_table->get_as<int64_t>("weight_quant")) {
        const int quant = static_cast<int>((*v).get());
        if (quant < 0 || quant > 2) {
            SPDLOG_ERROR("Invalid inference.ldnet.weight_quant {}; expected 0=FP32, 1=FP16, or 2=INT8.", quant);
            return EC::INVALID_COMMAND_ARGUMENTS;
        }
        ldnet_config.weight_quant = static_cast<NET_QUANTIZATION>(quant);
        ldnet_changed = true;
    }

    if (auto v = ldnet_table->get_as<int64_t>("input_width")) {
        const int width = static_cast<int>((*v).get());
        if (width <= 0) {
            SPDLOG_ERROR("Invalid inference.ldnet.input_width {}; expected > 0.", width);
            return EC::INVALID_COMMAND_ARGUMENTS;
        }
        ldnet_config.input_width = width;
        ldnet_changed = true;
    }

    if (auto v = ldnet_table->get_as<int64_t>("input_height")) {
        const int height = static_cast<int>((*v).get());
        if (height <= 0) {
            SPDLOG_ERROR("Invalid inference.ldnet.input_height {}; expected > 0.", height);
            return EC::INVALID_COMMAND_ARGUMENTS;
        }
        ldnet_config.input_height = height;
        ldnet_changed = true;
    }

    if (auto v = ldnet_table->get_as<bool>("embedded_nms")) {
        ldnet_config.embedded_nms = (*v).get();
        ldnet_changed = true;
    }

    if (auto v = ldnet_table->get_as<bool>("use_trt")) {
        ldnet_config.use_trt = (*v).get();
        ldnet_changed = true;
    }

    if (ldnet_changed) {
        inference_manager.SetLDNetConfig(ldnet_config.weight_quant,
                                         ldnet_config.input_width,
                                         ldnet_config.input_height,
                                         ldnet_config.embedded_nms,
                                         ldnet_config.use_trt);
    }

    LDNetConfig active_ldnet_config = inference_manager.GetLDNetConfig();
    SPDLOG_INFO("Inference config active: rc_version={} ld_version={} ldnet={}",
                inference_manager.GetRCVersion(),
                inference_manager.GetLDVersion(),
                active_ldnet_config.GetFileNameAppendix());
    return EC::OK;
}

void Configuration::ParseCameraCalibration()
{
    // Helper: read a 9-element TOML array into a 3×3 CV_64F matrix.
    // Returns an identity matrix if the key is missing or malformed.
    auto parse_mat3x3 = [](const toml::table& t, std::string_view key) -> cv::Mat {
        cv::Mat M = cv::Mat::eye(3, 3, CV_64F);
        const auto* arr = t[key].as_array();
        if (!arr || arr->size() != 9) {
            SPDLOG_WARN("Camera calibration: '{}' missing or not a 9-element array, using identity", key);
            return M;
        }
        for (int i = 0; i < 9; ++i)
            M.at<double>(i / 3, i % 3) = arr->at(i).value_or(0.0);
        return M;
    };

    // Shared intrinsics from [camera-calibration]
    const auto* calib_table = config["camera-calibration"].as_table();
    if (!calib_table) {
        SPDLOG_WARN("Camera calibration: [camera-calibration] section missing, using defaults");
    } else {
        camera_calibration.camera_matrix = parse_mat3x3(*calib_table, "camera_matrix");

        cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
        const auto* dist_arr = (*calib_table)["dist_coeffs"].as_array();
        if (!dist_arr || dist_arr->size() != 5) {
            SPDLOG_WARN("Camera calibration: 'dist_coeffs' missing or not a 5-element array, using zeros");
        } else {
            for (int i = 0; i < 5; ++i)
                D.at<double>(0, i) = dist_arr->at(i).value_or(0.0);
        }
        camera_calibration.dist_coeffs = D;
    }

    // Per-camera cam_to_body from each [camera-device.camN], indexed by the camera id field.
    for (const auto& [key, value] : *camera_devices_config)
    {
        const auto* cam_table = value.as_table();
        if (!cam_table) continue;
        const auto id_node = cam_table->get_as<int64_t>("id");
        if (!id_node) continue;
        const std::size_t cam_idx = static_cast<std::size_t>((*id_node).get());
        if (cam_idx >= NUM_CAMERAS) continue;
        camera_calibration.cam_to_body[cam_idx] = parse_mat3x3(*cam_table, "cam_to_body");
    }
}

template <typename T> T get_or_warn(const toml::table& t, std::string_view key, T def, std::string_view msg_key)
{
    if (auto v = t.get_as<T>(key))
        return (*v).get();
    SPDLOG_WARN("Failed to parse '{}' from camera config. Using default value: {}", msg_key, def);
    return def;
}
