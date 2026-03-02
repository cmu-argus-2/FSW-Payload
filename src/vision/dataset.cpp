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


Dataset::Dataset(double max_period, uint16_t nb_frames, 
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
    // TODO: may want to rethink this dataset naming approach, since dataset collection will start
    // with some delay from the creation of this
    folder_path = DATASETS_FOLDER + std::to_string(capture_start_time) + "/";

    bool res = DH::MakeNewDirectory(folder_path);
    if (!res)
    {
        SPDLOG_ERROR("Failed to create {}", folder_path);
    }

    CreateConfigurationFile(); 
}


Dataset::Dataset(const std::string& folder_path)
:
capture_start_time(timing::GetCurrentTimeMs()),
maximum_period(DEFAULT_COLLECTION_PERIOD), // default
target_frame_nb(MAX_SAMPLES), // default
dataset_capture_mode(CAPTURE_MODE::PERIODIC), // default
imu_collection_mode(IMU_COLLECTION_MODE::GYRO_MAG_TEMP),
image_capture_rate(60),
imu_sample_rate_hz(1.0f),
target_processing_stage(ProcessingStage::NotPrefiltered)
{
    
    std::string candidate_folder = folder_path;
    // Correct the folder path if needed
    if (candidate_folder.back() != '/')
    {
        candidate_folder += '/';
    }
    
    // check if config path exists
    if (!DH::fs::exists(candidate_folder + DATASET_CONFIG_FILE_NAME))
    {
        SPDLOG_ERROR("{} does not exist!", candidate_folder + DATASET_CONFIG_FILE_NAME);
        throw std::invalid_argument("The provided folder path does not exist..."); // throwing is the best way in this case
    }

    // read config file and fill all parameters
    // TODO: remove this high-level try-catch and throw OR have a default config... (well-documented)
    try
    {
        toml::table config = toml::parse_file(candidate_folder + DATASET_CONFIG_FILE_NAME);

        std::optional<double> max_period = config["maximum_period"].value<double>();
        if (!max_period)
        {
            throw std::invalid_argument("Missing or invalid 'maximum_period' in configuration.");
        }
        maximum_period = *max_period; // dereference the contained value
        if (maximum_period < ABSOLUTE_MINIMUM_PERIOD)
        {
            SPDLOG_ERROR("Maximum period {} is below the absolute minimum period {}", maximum_period, ABSOLUTE_MINIMUM_PERIOD);
            throw std::invalid_argument("Maximum period is below the absolute minimum period.");
        }

        if (maximum_period > ABSOLUTE_MAXIMUM_PERIOD)
        {
            SPDLOG_ERROR("Maximum period {} is above the absolute maximum period {}", maximum_period, ABSOLUTE_MAXIMUM_PERIOD);
            throw std::invalid_argument("Maximum period is above the absolute maximum period.");
        }

        std::optional<uint64_t> target_frames = config["target_frames"].value<uint64_t>();
        if (!target_frames)
        {
            throw std::invalid_argument("Missing or invalid 'target_frames' in configuration.");
        }
        target_frame_nb = static_cast<uint32_t>(*target_frames);
        if (target_frame_nb > MAX_SAMPLES)
        {
            SPDLOG_ERROR("Target frame number {} exceeds the maximum allowed {}", target_frame_nb, MAX_SAMPLES);
            throw std::invalid_argument("Target frame number exceeds the maximum allowed.");
        }

        std::optional<uint64_t> dataset_capture_mode_val = config["dataset_capture_mode"].value<uint64_t>();
        if (!dataset_capture_mode_val)
        {
            throw std::invalid_argument("Missing or invalid 'dataset_capture_mode' in configuration.");
        }
        dataset_capture_mode = static_cast<CAPTURE_MODE>(*dataset_capture_mode_val);
        if (!IsValidCaptureMode(dataset_capture_mode))
        {
            SPDLOG_ERROR("Invalid dataset capture mode {}", static_cast<int>(dataset_capture_mode));
            throw std::invalid_argument("Invalid dataset capture mode.");
        }
    }
    catch (const std::exception& e)
    {
        SPDLOG_ERROR("Failed to parse configuration file: {}", e.what());
        throw std::runtime_error("Failed to parse configuration file or incorrect configuration.");
    }

}



void Dataset::CreateConfigurationFile()
{
    auto tbl = toml::table{
        {"maximum_period", maximum_period},
        {"target_frames", target_frame_nb},
        {"dataset_capture_mode", dataset_capture_mode}
    };

    // Write to file
    std::ofstream file(folder_path + DATASET_CONFIG_FILE_NAME);
    if (file.is_open())
    {
        file << tbl;
        file.close();
        SPDLOG_INFO("Config file saved to: {}", folder_path + DATASET_CONFIG_FILE_NAME);
    }
    else
    {
        SPDLOG_ERROR("Failed to open file for writing");
    }
    // TODO: handle fail cases
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
    // IMU TODO: Store log file path for reference in dataset
    /*
    j["imu_log_file_path"] = imuManager.GetLogFile();
    // IMU number of time stamps collected
    uint64_t imu_line_count = 0;
    std::ifstream imu_file(imuManager.GetLogFile());
    std::string line;
    while (std::getline(imu_file, line))
    {
        imu_line_count++;
    }
    j["imu_timestamps_collected"] = imu_line_count;
    */
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

void Dataset::fromJson(const Json& j)
{
    folder_path = j.at("folder_path").get<std::string>();
    capture_start_time = j.at("capture_start_time").get<uint64_t>();
    maximum_period = j.at("maximum_period").get<double>();
    target_frame_nb = j.at("target_frame_nb").get<uint16_t>();
    dataset_capture_mode = static_cast<CAPTURE_MODE>(j.at("dataset_capture_mode").get<uint64_t>());
    imu_collection_mode = static_cast<IMU_COLLECTION_MODE>(j.at("imu_collection_mode").get<uint64_t>());
    image_capture_rate = j.at("image_capture_rate").get<uint8_t>();
    imu_sample_rate_hz = j.at("imu_sample_rate_hz").get<float>();
    target_processing_stage = static_cast<ProcessingStage>(j.at("target_processing_stage").get<uint64_t>());
    // TODO: Complete
}