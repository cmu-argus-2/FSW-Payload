/*
  run_od_pipeline <dataset_config_folder> [od_config_path] [system_config_path] [--out <out_path>]

  Captures a dataset from the dataset config, processes it to the requested
  stage, prepares OD measurements if needed, runs batch OD, and writes results.

    --out  File to write the generated results directory path into. Falls back
           to path.out if not provided or not writable.
*/

#include "configuration.hpp"
#include "core/timing.hpp"
#include "inference/inference_manager.hpp"
#include "navigation/od.hpp"
#include "vision/dataset.hpp"

#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "toml.hpp"

static constexpr const char* kDefaultODConfigPath     = "config/od.toml";
static constexpr const char* kDefaultSystemConfigPath = "config/config.toml";
static constexpr const char* kDefaultOutPath          = "path.out";

// Returns the provided path if writable, otherwise falls back to kDefaultOutPath.
static std::string ResolveOutPath(const std::string& path)
{
    if (path == kDefaultOutPath) return path;
    std::ofstream f(path, std::ios::app);
    if (f.is_open()) return path;
    spdlog::warn("Cannot write to '{}', using default output '{}'", path, kDefaultOutPath);
    return kDefaultOutPath;
}

static void WriteResult(const std::string& out_file, const std::string& content)
{
    std::ofstream f(out_file, std::ios::trunc);
    if (!f.is_open()) { spdlog::error("Failed to write result to '{}'", out_file); return; }
    f << content << '\n';
}

static EC ApplyInferenceCliOverrides(InferenceManager& inference_manager,
                                     const std::optional<int>& rc_version,
                                     const std::optional<int>& ld_version,
                                     const std::optional<int>& ld_weight_quant,
                                     const std::optional<int>& ld_input_width,
                                     const std::optional<int>& ld_input_height,
                                     const std::optional<bool>& ld_embedded_nms,
                                     const std::optional<bool>& ld_use_trt)
{
    if (rc_version) {
        EC ec = inference_manager.SetRCNetVersion(*rc_version);
        if (ec != EC::OK) return ec;
    }

    if (ld_version) {
        EC ec = inference_manager.SetLDNetVersion(*ld_version);
        if (ec != EC::OK) return ec;
    }

    LDNetConfig ldnet_config = inference_manager.GetLDNetConfig();
    bool ldnet_changed = false;

    if (ld_weight_quant) {
        ldnet_config.weight_quant = static_cast<NET_QUANTIZATION>(*ld_weight_quant);
        ldnet_changed = true;
    }
    if (ld_input_width) {
        ldnet_config.input_width = *ld_input_width;
        ldnet_changed = true;
    }
    if (ld_input_height) {
        ldnet_config.input_height = *ld_input_height;
        ldnet_changed = true;
    }
    if (ld_embedded_nms) {
        ldnet_config.embedded_nms = *ld_embedded_nms;
        ldnet_changed = true;
    }
    if (ld_use_trt) {
        ldnet_config.use_trt = *ld_use_trt;
        ldnet_changed = true;
    }

    if (ldnet_changed) {
        inference_manager.SetLDNetConfig(ldnet_config.weight_quant,
                                         ldnet_config.input_width,
                                         ldnet_config.input_height,
                                         ldnet_config.embedded_nms,
                                         ldnet_config.use_trt);
    }

    return EC::OK;
}

static bool LoadDatasetConfig(const std::string& folder_path, DatasetConfig& out)
{
    const std::string path = folder_path + "/dataset_config.toml";
    try {
        const toml::table cfg = toml::parse_file(path);
        out.maximum_period = cfg["maximum_period"].value_or(out.maximum_period);
        const uint64_t target_frame_nb =
            cfg["target_frame_nb"].value_or(uint64_t(out.target_frame_nb));
        if (target_frame_nb == 0 || target_frame_nb > MAX_SAMPLES) {
            spdlog::error("Invalid target_frame_nb {} in {}", target_frame_nb, path);
            return false;
        }
        out.target_frame_nb = static_cast<uint8_t>(target_frame_nb);
        out.capture_mode = static_cast<CAPTURE_MODE>(
            cfg["dataset_capture_mode"].value_or(uint64_t(out.capture_mode)));
        out.imu_collection_mode = static_cast<IMU_COLLECTION_MODE>(
            cfg["imu_collection_mode"].value_or(uint64_t(out.imu_collection_mode)));
        out.image_capture_rate = static_cast<uint8_t>(
            cfg["image_capture_rate"].value_or(uint64_t(out.image_capture_rate)));
        out.imu_sample_rate_hz = static_cast<float>(
            cfg["imu_sample_rate_hz"].value_or(double(out.imu_sample_rate_hz)));
        out.target_processing_stage = static_cast<ProcessingStage>(
            cfg["target_processing_stage"].value_or(uint64_t(out.target_processing_stage)));
        if (const auto* arr = cfg["active_cameras"].as_array())
            for (size_t i = 0; i < NUM_CAMERAS && i < arr->size(); ++i)
                if (auto val = (*arr)[i].value<bool>()) out.active_cameras[i] = *val;
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse dataset config {}: {}", path, e.what());
        return false;
    }

    if (out.capture_start_time == 0) {
        out.capture_start_time = timing::GetCurrentTimeMs();
    }
    return Dataset::isValidConfiguration(out.maximum_period,
                                         out.target_frame_nb,
                                         out.capture_mode,
                                         out.imu_collection_mode,
                                         out.image_capture_rate,
                                         out.imu_sample_rate_hz,
                                         out.target_processing_stage,
                                         out.capture_start_time);
}

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::info);

    CLI::App app{"Capture a dataset, prepare measurements, run OD, and write results."};
    app.allow_extras(false);

    std::string dataset_config_folder;
    std::string od_config_path = kDefaultODConfigPath;
    std::string system_config_path = kDefaultSystemConfigPath;
    std::string out_arg = kDefaultOutPath;
    std::optional<int> rc_version;
    std::optional<int> ld_version;
    std::optional<int> ld_weight_quant;
    std::optional<int> ld_input_width;
    std::optional<int> ld_input_height;
    bool ld_embedded_nms = false;
    bool ld_use_trt = false;

    app.add_option("dataset_config_folder", dataset_config_folder,
                   "Folder containing dataset_config.toml")->required();
    app.add_option("od_config_path", od_config_path,
                   "OD config TOML (default: config/od.toml)");
    app.add_option("system_config_path", system_config_path,
                   "System config TOML (default: config/config.toml)");
    app.add_option("--out", out_arg, "File to write the generated results path into");
    app.add_option("--rc-version", rc_version, "RCNet model version");
    app.add_option("--ld-version", ld_version, "LDNet model version");
    app.add_option("--ld-weight-quant", ld_weight_quant, "LDNet weight quantization: 0=FP32 1=FP16 2=INT8")
        ->check(CLI::Range(0, 2));
    app.add_option("--ld-input-width", ld_input_width, "LDNet input width in pixels")
        ->check(CLI::PositiveNumber);
    app.add_option("--ld-input-height", ld_input_height, "LDNet input height in pixels")
        ->check(CLI::PositiveNumber);
    CLI::Option* ld_embedded_nms_opt =
        app.add_flag("--ld-embedded-nms,--no-ld-embedded-nms{false}", ld_embedded_nms,
                     "Use LDNet engines with embedded NMS");
    CLI::Option* ld_use_trt_opt =
        app.add_flag("--ld-use-trt,--no-ld-use-trt{false}", ld_use_trt,
                     "Use TensorRT LDNet engines instead of ONNX");

    CLI11_PARSE(app, argc, argv);

    const std::string out_path = ResolveOutPath(out_arg);

    ODRequest request;
    if (!LoadDatasetConfig(dataset_config_folder, request.dataset_config)) {
        return 1;
    }
    request.od_config_path     = od_config_path;
    request.system_config_path = system_config_path;

    auto config = std::make_unique<Configuration>();
    try {
        config->LoadConfiguration(request.system_config_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load system configuration {}: {}",
                      request.system_config_path, e.what());
        return 1;
    }

    InferenceManager inference_manager;
    EC ec = config->ApplyInferenceConfig(inference_manager);
    if (ec != EC::OK) return to_uint8(ec);

    const std::optional<bool> ld_embedded_nms_override =
        ld_embedded_nms_opt->count() > 0 ? std::optional<bool>(ld_embedded_nms) : std::nullopt;
    const std::optional<bool> ld_use_trt_override =
        ld_use_trt_opt->count() > 0 ? std::optional<bool>(ld_use_trt) : std::nullopt;

    ec = ApplyInferenceCliOverrides(inference_manager,
                                    rc_version,
                                    ld_version,
                                    ld_weight_quant,
                                    ld_input_width,
                                    ld_input_height,
                                    ld_embedded_nms_override,
                                    ld_use_trt_override);
    if (ec != EC::OK) return to_uint8(ec);

    IMUManager imu_manager(config->GetIMUConfig());
    CameraManager camera_manager(config->GetCameraConfigs(),
                                 config->GetCameraISPConfig(),
                                 inference_manager);

    std::thread imu_thread(&IMUManager::RunLoop, &imu_manager);
    [[maybe_unused]] int nb_enabled_cams = camera_manager.EnableCameras();
    std::thread camera_thread(&CameraManager::RunLoop, &camera_manager);

    auto stop_threads = [&]() {
        imu_manager.StopLoop();
        if (imu_thread.joinable()) imu_thread.join();
        camera_manager.StopLoops();
        if (camera_thread.joinable()) camera_thread.join();
    };

    const ODResult result = RunODPipeline(request, camera_manager, imu_manager, inference_manager);
    stop_threads();

    if (result.code != ErrorCode::OK) {
        spdlog::error("OD pipeline failed at stage {} with error code {}.",
                      static_cast<int>(result.stage), static_cast<int>(result.code));
        return 1;
    }

    spdlog::info("OD complete. Dataset {} results in {}",
                 result.dataset_folder, result.results_dir);
    WriteResult(out_path, result.results_dir);
    return 0;
}
