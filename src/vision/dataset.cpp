#include <ostream>
#include <stdexcept>
#include "toml.hpp"
#include "spdlog/spdlog.h"

#include "vision/dataset.hpp"
#include "core/data_handling.hpp"
#include "core/timing.hpp"


std::vector<std::string> Dataset::ListAllStoredDatasets()
{
    // List all folders in the dataset folder
    std::vector<std::string> dataset_folders;
    for (const auto& entry : std::filesystem::directory_iterator(DATASETS_FOLDER))
    {
        if (entry.is_directory())
        {
            dataset_folders.push_back(DATASETS_FOLDER + entry.path().filename().string());
        }
    }
    return dataset_folders;
}

bool Dataset::isValidConfigurationFile(const std::string& config_file_path)
{
    toml::table config = toml::parse_file(config_file_path);
    
    std::optional<double> max_period = config["maximum_period"].value<double>();
    if (!max_period)
    {
        SPDLOG_ERROR("Missing or invalid 'maximum_period' in configuration.");
        return false;
    }

    std::optional<uint64_t> target_frames = config["target_frame_nb"].value<uint64_t>();
    if (!target_frames)
    {
        SPDLOG_ERROR("Missing or invalid 'target_frame_nb' in configuration.");
        return false;
    }

    std::optional<uint64_t> dataset_capture_mode_val = config["dataset_capture_mode"].value<uint64_t>();
    if (!dataset_capture_mode_val)
    {
        SPDLOG_ERROR("Missing or invalid 'dataset_capture_mode' in configuration.");
        return false;
    }

    // imu collection mode
    std::optional<uint64_t> imu_collection_mode_val = config["imu_collection_mode"].value<uint64_t>();
    if (!imu_collection_mode_val)
    {
        SPDLOG_ERROR("Missing or invalid 'imu_collection_mode' in configuration.");
        return false;
    }

    // image capture rate
    std::optional<uint64_t> image_capture_rate_val = config["image_capture_rate"].value<uint64_t>();
    if (!image_capture_rate_val)    {
        SPDLOG_ERROR("Missing or invalid 'image_capture_rate' in configuration.");
        return false;
    }

    // imu sample rate
    std::optional<double> imu_sample_rate_hz_val = config["imu_sample_rate_hz"].value<double>();
    if (!imu_sample_rate_hz_val)    {
        SPDLOG_ERROR("Missing or invalid 'imu_sample_rate_hz' in configuration.");
        return false;
    }

    // target processing stage
    std::optional<uint64_t> target_processing_stage_val = config["target_processing_stage"].value<uint64_t>();
    if (!target_processing_stage_val)    {
        SPDLOG_ERROR("Missing or invalid 'target_processing_stage' in configuration.");
        return false;
    }

    return isValidConfiguration(*max_period, static_cast<uint8_t>(*target_frames), 
                                static_cast<CAPTURE_MODE>(*dataset_capture_mode_val),
                                static_cast<IMU_COLLECTION_MODE>(*imu_collection_mode_val),
                                static_cast<uint8_t>(*image_capture_rate_val),
                                static_cast<float>(*imu_sample_rate_hz_val),
                                static_cast<ProcessingStage>(*target_processing_stage_val),
                                timing::GetCurrentTimeMs());
}

bool Dataset::isValidConfiguration(double max_period, uint8_t nb_frames, CAPTURE_MODE capture_mode, IMU_COLLECTION_MODE imu_collection_mode,
                                    uint8_t image_capture_rate, float imu_sample_rate_hz, ProcessingStage target_processing_stage,
                                    uint64_t capture_start_time)
{
    if (max_period < ABSOLUTE_MINIMUM_PERIOD)
    {
        SPDLOG_ERROR("Maximum period {} is below the absolute minimum period {}", max_period, ABSOLUTE_MINIMUM_PERIOD);
        return false;
    }

    if (max_period > ABSOLUTE_MAXIMUM_PERIOD)
    {
        SPDLOG_ERROR("Maximum period {} is above the absolute maximum period {}", max_period, ABSOLUTE_MAXIMUM_PERIOD);
        return false;
    }
    
    if (nb_frames > MAX_SAMPLES || nb_frames <= 0)
    {
        SPDLOG_ERROR("Target frame number {} exceeds the maximum allowed {}", nb_frames, MAX_SAMPLES);
        return false;
    }

    if (!IsValidCaptureMode(capture_mode))
    {
        SPDLOG_ERROR("Invalid dataset capture mode {}", static_cast<int>(capture_mode));
        return false;
    }

    if (imu_collection_mode > IMU_COLLECTION_MODE::GYRO_MAG_TEMP || imu_collection_mode < IMU_COLLECTION_MODE::NONE)
    {
        SPDLOG_ERROR("Invalid IMU collection mode {}", static_cast<int>(imu_collection_mode));
        return false;
    }

    if (image_capture_rate <= 0)
    {
        SPDLOG_ERROR("Image capture rate cannot be leq zero");
        return false;
    }

    if (imu_sample_rate_hz <= 0 || imu_sample_rate_hz > 25.0)
    {
        SPDLOG_ERROR("IMU sample rate outside allowed bounds");
        return false;
    }

    if (target_processing_stage < ProcessingStage::NotPrefiltered || target_processing_stage > ProcessingStage::LDNeted)
    {
        SPDLOG_ERROR("Invalid target processing stage {}", static_cast<int>(target_processing_stage));
        return false;
    }

    return true;
}


Dataset::Dataset(double max_period, uint8_t nb_frames, 
                CAPTURE_MODE capture_mode,
                IMU_COLLECTION_MODE imu_collection_mode,
                uint8_t image_capture_rate, float imu_sample_rate_hz, 
                ProcessingStage target_processing_stage,
                uint64_t capture_start_time=timing::GetCurrentTimeMs())
:
capture_start_time(capture_start_time),
maximum_period(max_period),
target_frame_nb(nb_frames),
dataset_capture_mode(capture_mode),
imu_collection_mode(imu_collection_mode),
image_capture_rate(image_capture_rate),
imu_sample_rate_hz(imu_sample_rate_hz),
target_processing_stage(target_processing_stage)
{
    if (!isValidConfiguration(max_period, nb_frames, capture_mode, imu_collection_mode,
                                image_capture_rate, imu_sample_rate_hz, 
                                target_processing_stage, capture_start_time))
    {
        throw std::invalid_argument("Invalid dataset configuration parameters.");
    }
    // TODO: may want to rethink this dataset naming approach, since dataset collection will start
    // with some delay from the creation of this
    folder_path = DATASETS_FOLDER + std::to_string(capture_start_time) + "/";

    bool res = DH::MakeNewDirectory(folder_path);
    if (!res)
    {
        SPDLOG_ERROR("Failed to create {}", folder_path);
    }

    CreateConfigurationFile(); 
    imu_log_file_path = folder_path + "imu_data.csv";
}


Dataset::Dataset(const std::string& folder_path_in)
:
capture_start_time(timing::GetCurrentTimeMs()),
maximum_period(DEFAULT_COLLECTION_PERIOD), // default
target_frame_nb(MAX_SAMPLES), // default
dataset_capture_mode(CAPTURE_MODE::PERIODIC), // default
imu_collection_mode(IMU_COLLECTION_MODE::GYRO_ONLY),
image_capture_rate(60),
imu_sample_rate_hz(1.0f),
target_processing_stage(ProcessingStage::NotPrefiltered),
folder_path(folder_path_in),
imu_log_file_path(folder_path_in + "/imu_data.csv")
{
    
    std::string candidate_folder = folder_path_in;
    // Correct the folder path if needed
    if (candidate_folder.back() != '/')
    {
        candidate_folder += '/';
    }
    // check if config path exists
    if (!DH::fs::exists(candidate_folder + DATASET_CONFIG_FILE_NAME))
    {
        SPDLOG_ERROR("{} does not exist!", candidate_folder + DATASET_CONFIG_FILE_NAME);
        throw std::invalid_argument("Dataset configuration file not found.");
    }
    // read config file and fill all parameters
    // TODO: remove this high-level try-catch and throw OR have a default config... (well-documented)
    if (!isValidConfigurationFile(candidate_folder + DATASET_CONFIG_FILE_NAME))
    {
        throw std::invalid_argument("Dataset configuration file is invalid.");
    }
    // TODO: TOML will only have the initial configuration parameters
    // To load with results, it should be done with json or the toml should be updated
    // in which case the other would become redundant
    // initializing a dataset through a toml could still be done for predefined configurations, 
    // but then this constructor should receive the toml path, not the folder
    toml::table config = toml::parse_file(candidate_folder + DATASET_CONFIG_FILE_NAME);

    // If not available, default value is kept
    if (config.contains("maximum_period")) {
        maximum_period      = *(config["maximum_period"].value<double>());
    }
    target_frame_nb         = static_cast<uint16_t>(*(config["target_frame_nb"].value<uint64_t>()));
    dataset_capture_mode    = static_cast<CAPTURE_MODE>(*(config["dataset_capture_mode"].value<uint64_t>()));
    imu_collection_mode     = static_cast<IMU_COLLECTION_MODE>(*(config["imu_collection_mode"].value<uint64_t>()));
    image_capture_rate      = static_cast<uint8_t>(*(config["image_capture_rate"].value<uint64_t>()));
    imu_sample_rate_hz      = static_cast<float>(*(config["imu_sample_rate_hz"].value<double>()));
    target_processing_stage = static_cast<ProcessingStage>(*(config["target_processing_stage"].value<uint64_t>())); // Use uint64_t to match the value type in config
    capture_start_time = static_cast<uint64_t>(*(config["capture_start_time"].value<int64_t>()));
}

Dataset& Dataset::operator=(const Dataset& other)
{
    if (this != &other) {
        folder_path             = other.folder_path; 
        imu_log_file_path       = other.imu_log_file_path;
        capture_start_time      = other.capture_start_time;
        maximum_period          = other.maximum_period;
        target_frame_nb         = other.target_frame_nb;
        dataset_capture_mode    = other.dataset_capture_mode;
        imu_collection_mode     = other.imu_collection_mode;
        image_capture_rate      = other.image_capture_rate;
        target_processing_stage = other.target_processing_stage;
        stored_frame_ids        = other.stored_frame_ids;
    }
    return *this;
}


bool Dataset::CreateConfigurationFile()
{
    auto tbl = toml::table{
        {"capture_start_time", static_cast<int64_t>(capture_start_time)}, 
        {"maximum_period", maximum_period},
        {"target_frames", target_frame_nb},
        {"dataset_capture_mode", dataset_capture_mode},
        {"imu_collection_mode", imu_collection_mode},
        {"image_capture_rate", image_capture_rate},
        {"imu_sample_rate_hz", imu_sample_rate_hz},
        {"target_processing_stage", target_processing_stage}
    };

    // Write to file
    std::ofstream file(folder_path + DATASET_CONFIG_FILE_NAME, std::ofstream::out | std::ofstream::trunc);
    if (file.is_open())
    {
        file << tbl;
        file.close();
        SPDLOG_INFO("Config file saved to: {}", folder_path + DATASET_CONFIG_FILE_NAME);
    }
    else
    {
        SPDLOG_ERROR("Failed to open file for writing");
        return false;
    }
    return true;
}


Json Dataset::toJson() const
{
    Json j;
    // Dataset Configuration
    j["folder_path"] = folder_path;
    j["capture_start_time"] = capture_start_time;
    j["maximum_period"] = maximum_period;
    j["target_frame_nb"] = target_frame_nb;
    j["dataset_capture_mode"] = dataset_capture_mode;
    j["imu_collection_mode"] = imu_collection_mode;
    j["image_capture_rate"] = image_capture_rate;
    j["imu_sample_rate_hz"] = imu_sample_rate_hz;
    j["target_processing_stage"] = target_processing_stage;
    // IMU
    j["imu_log_file_path"] = imu_log_file_path;
    // IMU number of timestamps collected
    uint64_t imu_line_count = 0;
    std::ifstream imu_file(imu_log_file_path);
    std::string line;
    while (std::getline(imu_file, line))
    {
        imu_line_count++;
    }
    j["imu_timestamps_collected"] = imu_line_count;
    
    // Number of frames stored in the dataset
    j["frames_collected"] = stored_frame_ids.size();
    j["frame_id_list"] = stored_frame_ids; // list of frame ids (cam_id, timestamp) stored in the dataset
    int num_frames_prefiltered = 0;
    int num_frames_rcneted = 0;
    int num_frames_ldneted = 0;
    int num_frames_earth = 0;
    int num_frames_roi = 0;
    int num_frames_landmarks = 0;
     for (const auto& frame_id : stored_frame_ids)
    {
        // Load frame metadata from disk using the frame_id (cam_id and timestamp)
        // Check if the frame meets the criteria for each category and increment the corresponding counters
        Json frame_metadata = DH::LoadFrameMetadataFromDisk(std::get<1>(frame_id), std::get<0>(frame_id));
        if (frame_metadata.is_null()) {
            SPDLOG_WARN("Failed to load metadata for frame ID: ({}, {})", std::get<0>(frame_id), std::get<1>(frame_id));
            continue;
        }
        if (frame_metadata.contains("processing_stage"))
        {
            int processing_stage = frame_metadata["processing_stage"].get<int>();
            if (processing_stage >= 1) num_frames_prefiltered++;
            if (processing_stage >= 2) num_frames_rcneted++;
            if (processing_stage >= 3) num_frames_ldneted++;
        }
        if (frame_metadata.contains("annotation_state"))
        {
            int annotation_state = frame_metadata["annotation_state"].get<int>();
            if (annotation_state >= 1) num_frames_earth++;
            if (annotation_state >= 2) num_frames_roi++;
            if (annotation_state >= 3) num_frames_landmarks++;
        }
    }
    j["num_frames_prefiltered"] = num_frames_prefiltered;
    j["num_frames_rcneted"] = num_frames_rcneted;
    j["num_frames_ldneted"] = num_frames_ldneted;
    j["num_frames_earth"] = num_frames_earth;
    j["num_frames_roi"] = num_frames_roi;
    j["num_frames_landmarks"] = num_frames_landmarks;
    
    return j;
}

bool Dataset::fromJson(const Json& j)
{
    if (!j.contains("folder_path") || !j.contains("capture_start_time") || !j.contains("maximum_period") || 
        !j.contains("target_frame_nb") || !j.contains("dataset_capture_mode") || !j.contains("imu_collection_mode") ||
        !j.contains("image_capture_rate") || !j.contains("imu_sample_rate_hz") || !j.contains("target_processing_stage")
        || !j.contains("imu_log_file_path") || !j.contains("frame_id_list"))
    {
        SPDLOG_ERROR("JSON object is missing required fields to construct a Dataset.");
        return false;
    }
    folder_path = j.at("folder_path").get<std::string>();
    imu_log_file_path = j.at("imu_log_file_path").get<std::string>();
    capture_start_time = j.at("capture_start_time").get<uint64_t>();
    maximum_period = j.at("maximum_period").get<double>();
    target_frame_nb = j.at("target_frame_nb").get<uint16_t>();
    dataset_capture_mode = static_cast<CAPTURE_MODE>(j.at("dataset_capture_mode").get<uint64_t>());
    imu_collection_mode = static_cast<IMU_COLLECTION_MODE>(j.at("imu_collection_mode").get<uint64_t>());
    image_capture_rate = j.at("image_capture_rate").get<uint8_t>();
    imu_sample_rate_hz = j.at("imu_sample_rate_hz").get<float>();
    target_processing_stage = static_cast<ProcessingStage>(j.at("target_processing_stage").get<uint64_t>());
    stored_frame_ids = j.at("frame_id_list").get<std::vector<std::tuple<uint8_t, uint64_t>>>();

    return true;
}


bool Dataset::OverlapsWith(const Dataset& other) const
{
    return (capture_start_time >= other.capture_start_time && capture_start_time <= other.capture_start_time + other.maximum_period * 1000) ||
           (capture_start_time + maximum_period * 1000 >= other.capture_start_time && capture_start_time + maximum_period * 1000 <= other.capture_start_time + other.maximum_period * 1000);
}