/*
    Test file to replicate what should happen in FSW when the command to start 
    dataset collection is received.
*/
#include "spdlog/spdlog.h"
#include "vision/dataset_manager.hpp"
#include "inference/inference_manager.hpp"
#include "core/timing.hpp"
#include "configuration.hpp"
#include <memory>
#include <thread>
#include <array>
#include <getopt.h>
#include <iostream>
#include <cstdlib>

#define DATASET_KEY_CMD "CMD"


static void PrintUsage(const char* prog)
{
    std::cerr <<
        "Usage: " << prog << " [OPTIONS]\n"
        "\n"
        "Path options:\n"
        "  -c, --config <path>         Main config file           (default: config/config.toml)\n"
        "  -d, --ds-config-dir <path>  Dataset config folder      (default: config)\n"
        "\n"
        "Dataset parameters (override dataset_config.toml):\n"
        "  -m, --max-period <secs>     Maximum collection period  (default: from config)\n"
        "  -n, --frame-nb <count>      Target frame count         (default: from config)\n"
        "  -M, --capture-mode <int>    Capture mode enum value    (default: from config)\n"
        "  -I, --imu-mode <int>        IMU collection mode enum   (default: from config)\n"
        "  -r, --image-rate <int>      Image capture rate         (default: from config)\n"
        "  -s, --imu-rate <hz>         IMU sample rate (Hz)       (default: from config)\n"
        "  -p, --proc-stage <int>      Processing stage enum      (default: from config)\n"
        "  -t, --capture-start-time <ms> Capture start timestamp (ms since epoch) (default: current time)\n"
        "\n"
        "  -h, --help                  Show this help and exit\n";
}

static CAPTURE_MODE ParseCaptureMode(const char* s)
{
    SPDLOG_INFO("Parsing capture mode: {}", s);
    if (std::string(s) == "IDLE")           return CAPTURE_MODE::IDLE;
    if (std::string(s) == "CAPTURE_SINGLE") return CAPTURE_MODE::CAPTURE_SINGLE;
    if (std::string(s) == "PERIODIC")       return CAPTURE_MODE::PERIODIC;
    if (std::string(s) == "PERIODIC_EARTH") return CAPTURE_MODE::PERIODIC_EARTH;
    if (std::string(s) == "PERIODIC_ROI")   return CAPTURE_MODE::PERIODIC_ROI;
    if (std::string(s) == "PERIODIC_LDMK")  return CAPTURE_MODE::PERIODIC_LDMK;
    throw std::invalid_argument(std::string("Unknown capture mode: ") + s);
}

int main(int argc, char** argv)
{
    std::string config_file_path = "config/config.toml";
    std::string ds_config_folder_path = "config";

    double   opt_max_period      = -1.0;
    int      opt_target_frame_nb = -1;
    int      opt_capture_mode    = -1;
    int      opt_imu_mode        = -1;
    int      opt_image_rate      = -1;
    double   opt_imu_rate_hz     = -1.0;
    int      opt_proc_stage      = -1;
    int64_t  opt_capture_start_time = -1;

    static const char* short_opts = "c:d:m:n:M:I:r:s:p:t:h";
    static const struct option long_opts[] = {
        { "config",        required_argument, nullptr, 'c' },
        { "ds-config-dir", required_argument, nullptr, 'd' },
        { "max-period",    required_argument, nullptr, 'm' },
        { "frame-nb",      required_argument, nullptr, 'n' },
        { "capture-mode",  required_argument, nullptr, 'M' },
        { "imu-mode",      required_argument, nullptr, 'I' },
        { "image-rate",    required_argument, nullptr, 'r' },
        { "imu-rate",         required_argument, nullptr, 's' },
        { "proc-stage",       required_argument, nullptr, 'p' },
        { "capture-start-time", required_argument, nullptr, 't' },
        { "help",             no_argument,       nullptr, 'h' },
        { nullptr,         0,                 nullptr,  0  }
    };

    int opt;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1)
    {   
        switch (opt)
        {
            case 'c': config_file_path       = optarg;              break;
            case 'd': ds_config_folder_path  = optarg;              break;
            case 'm': opt_max_period         = std::atof(optarg);   break;
            case 'n': opt_target_frame_nb    = std::atoi(optarg);   break;
            case 'M': opt_capture_mode = static_cast<int>(ParseCaptureMode(optarg)); break;
            case 'I': opt_imu_mode           = std::atoi(optarg);   break;
            case 'r': opt_image_rate         = std::atoi(optarg);   break;
            case 's': opt_imu_rate_hz        = std::atof(optarg);   break;
            case 'p': opt_proc_stage         = std::atoi(optarg);   break;
            case 't': opt_capture_start_time = std::atoll(optarg);  break;
            case 'h': PrintUsage(argv[0]); return 0;
            default:  PrintUsage(argv[0]); return 1;
        }
    }

    // Dataset collection configuration flag
    auto config = std::make_unique<Configuration>();
    try {
        config->LoadConfiguration(config_file_path);
    } catch (const toml::parse_error& err) {
        std::cerr << "Parsing configuration file failed: " << err << "\n";
        return 1;
    }
    SPDLOG_INFO("Configuration file {} loaded.", config_file_path);

    CAPTURE_MODE capture_mode = CAPTURE_MODE::PERIODIC;
    IMU_COLLECTION_MODE imu_collection_mode = IMU_COLLECTION_MODE::GYRO_MAG_TEMP;
    double max_period = 10.0;
    uint16_t target_frame_nb = 4;
    uint8_t image_capture_rate = uint8_t(1);
    float imu_sample_rate_hz = 1.0f;
    ProcessingStage target_processing_stage = ProcessingStage::NotPrefiltered;

    // Load dataset parameters from config file
    const std::string ds_config_path = ds_config_folder_path + "/dataset_config.toml";
    try {
        toml::table ds_cfg = toml::parse_file(ds_config_path);
        max_period              = ds_cfg["maximum_period"].value_or(max_period);
        target_frame_nb         = static_cast<uint16_t>(ds_cfg["target_frame_nb"].value_or(uint64_t(target_frame_nb)));
        capture_mode            = static_cast<CAPTURE_MODE>(ds_cfg["dataset_capture_mode"].value_or(uint64_t(capture_mode)));
        imu_collection_mode     = static_cast<IMU_COLLECTION_MODE>(ds_cfg["imu_collection_mode"].value_or(uint64_t(imu_collection_mode)));
        image_capture_rate      = static_cast<uint8_t>(ds_cfg["image_capture_rate"].value_or(uint64_t(image_capture_rate)));
        imu_sample_rate_hz      = static_cast<float>(ds_cfg["imu_sample_rate_hz"].value_or(double(imu_sample_rate_hz)));
        target_processing_stage = static_cast<ProcessingStage>(ds_cfg["target_processing_stage"].value_or(uint64_t(target_processing_stage)));
    } catch (const toml::parse_error& err) { 
        spdlog::error("Failed to parse dataset config {}: {}", ds_config_path, err.description());
        return 1;
    }

    if (opt_max_period      >= 0.0) max_period              = opt_max_period;
    if (opt_target_frame_nb >= 0)   target_frame_nb         = static_cast<uint16_t>(opt_target_frame_nb);
    if (opt_capture_mode    >= 0)   capture_mode            = static_cast<CAPTURE_MODE>(opt_capture_mode);
    if (opt_imu_mode        >= 0)   imu_collection_mode     = static_cast<IMU_COLLECTION_MODE>(opt_imu_mode);
    if (opt_image_rate      >= 0)   image_capture_rate      = static_cast<uint8_t>(opt_image_rate);
    if (opt_imu_rate_hz     >= 0.0) imu_sample_rate_hz      = static_cast<float>(opt_imu_rate_hz);
    if (opt_proc_stage      >= 0)   target_processing_stage = static_cast<ProcessingStage>(opt_proc_stage);

    SPDLOG_INFO("Parameters: max_period={} target_frame_nb={} capture_mode={} imu_mode={} image_rate={} imu_rate={} proc_stage={}",
    max_period, target_frame_nb, static_cast<uint8_t>(capture_mode),
    static_cast<uint8_t>(imu_collection_mode), image_capture_rate,
    imu_sample_rate_hz, static_cast<uint8_t>(target_processing_stage));

    // capture_start_time = 0 in config means "start immediately"
    int64_t capture_start_time = timing::GetCurrentTimeMs();
    if (opt_capture_start_time >= 0) capture_start_time = opt_capture_start_time;

    // collect IMU data or not flag
    const auto& imu_config = config->GetIMUConfig();
    IMUManager imu_manager(imu_config);

    InferenceManager inference_manager;

    const auto& cam_configs = config->GetCameraConfigs();
    CameraManager camera_manager(cam_configs, inference_manager);

    std::thread imu_thread = std::thread(&IMUManager::RunLoop, &imu_manager);

    std::array<bool, NUM_CAMERAS> temp;
    [[maybe_unused]] int nb_enabled_cams = camera_manager.EnableCameras(temp);
    std::thread camera_thread = std::thread(&CameraManager::RunLoop, &camera_manager);

    // how to make the above accessible to the dataset manager through the sys namespace?

    if (max_period == 0.0 || target_frame_nb == 0)
    {
        spdlog::error("Invalid parameters for dataset collection command");
        return 0;
    }

    auto ds = DatasetManager::GetActiveDatasetManager(DATASET_KEY_CMD);

    if (ds) // if already exists
    {
        // need to ensure it's actually running
        if (ds->Running())
        {
            // if running: TODO: return ERROR ACK saying that a dataset is already running
            // If completed, stop it then too
            SPDLOG_ERROR("Dataset already running under key {}, ignoring command", DATASET_KEY_CMD);
            imu_manager.StopLoop();
            if (imu_thread.joinable()) imu_thread.join();
            camera_manager.StopLoops();
            if (camera_thread.joinable()) camera_thread.join();
            return 1;
        }
        else
        {
            ds->StopDatasetManager(DATASET_KEY_CMD); // remove it (will create a new one)
        }
    }

    // Create a new Dataset
    SPDLOG_INFO("Starting dataset collection (type {}) for {} frames at a period of {} seconds.", static_cast<uint8_t>(capture_mode), target_frame_nb, max_period);

    ds = DatasetManager::Create(max_period, target_frame_nb, capture_mode, capture_start_time,
                                imu_collection_mode, image_capture_rate, imu_sample_rate_hz,
                                target_processing_stage, DATASET_KEY_CMD, camera_manager, imu_manager, inference_manager);
    ds->StartCollection();

    // StartCollection is asynchronous — block here until the collection loop
    // finishes naturally (frame target met or period elapsed).
    while (ds->Running())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Close Dataset Manager (joins the collection thread internally)
    ds->StopDatasetManager(DATASET_KEY_CMD);

    // close threads
    imu_manager.StopLoop();
    if (imu_thread.joinable()) imu_thread.join();

    camera_manager.StopLoops();
    if (camera_thread.joinable()) camera_thread.join();

    return 0;
}