#include <algorithm>
#include <filesystem>
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

// Validates raw (pre-cast) dataset parameter values against the same rules as
// isValidConfiguration and isValidConfigurationFile. Called by both to avoid
// duplicating the range checks.
bool Dataset::validateRawParams(double max_period, uint64_t target_frame_nb,
                                uint64_t capture_mode, uint64_t imu_mode,
                                uint64_t image_rate, float imu_rate,
                                uint64_t proc_stage)
{
    if (max_period < ABSOLUTE_MINIMUM_PERIOD || max_period > ABSOLUTE_MAXIMUM_PERIOD)
    {
        SPDLOG_ERROR("'maximum_period' value {} is out of range [{}, {}]",
                     max_period, ABSOLUTE_MINIMUM_PERIOD, ABSOLUTE_MAXIMUM_PERIOD);
        return false;
    }
    if (target_frame_nb == 0 || target_frame_nb > MAX_SAMPLES)
    {
        SPDLOG_ERROR("'target_frame_nb' value {} is out of range [1, {}]", target_frame_nb, MAX_SAMPLES);
        return false;
    }
    if (capture_mode > static_cast<uint64_t>(CAPTURE_MODE::PERIODIC_LDMK) ||
        !IsValidCaptureMode(static_cast<CAPTURE_MODE>(capture_mode)))
    {
        SPDLOG_ERROR("'dataset_capture_mode' value {} is not a valid capture mode", capture_mode);
        return false;
    }
    if (imu_mode > static_cast<uint64_t>(IMU_COLLECTION_MODE::GYRO_MAG_TEMP))
    {
        SPDLOG_ERROR("'imu_collection_mode' value {} is out of range", imu_mode);
        return false;
    }
    if (image_rate == 0 || image_rate > MAX_SAMPLES)
    {
        SPDLOG_ERROR("'image_capture_rate' value {} is out of range [1, {}]", image_rate, MAX_SAMPLES);
        return false;
    }
    if (imu_rate <= 0.0f || imu_rate > 25.0f)
    {
        SPDLOG_ERROR("'imu_sample_rate_hz' value {} is out of range (0, 25]", imu_rate);
        return false;
    }
    if (proc_stage > static_cast<uint64_t>(ProcessingStage::LDNeted))
    {
        SPDLOG_ERROR("'target_processing_stage' value {} is out of range", proc_stage);
        return false;
    }
    return true;
}

bool Dataset::isValidConfigurationFile(const std::string& config_file_path)
{
    toml::table config;
    try {
        config = toml::parse_file(config_file_path);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to parse configuration file '{}': {}", config_file_path, e.what());
        return false;
    }

    // Read all fields and verify they are present with the right type.
    std::optional<double>   max_period              = config["maximum_period"].value<double>();
    std::optional<uint64_t> target_frames           = config["target_frame_nb"].value<uint64_t>();
    std::optional<uint64_t> dataset_capture_mode_val= config["dataset_capture_mode"].value<uint64_t>();
    std::optional<uint64_t> imu_collection_mode_val = config["imu_collection_mode"].value<uint64_t>();
    std::optional<uint64_t> image_capture_rate_val  = config["image_capture_rate"].value<uint64_t>();
    std::optional<double>   imu_sample_rate_hz_val  = config["imu_sample_rate_hz"].value<double>();
    std::optional<uint64_t> target_processing_stage_val = config["target_processing_stage"].value<uint64_t>();
    std::optional<int64_t>  capture_start_time_val  = config["capture_start_time"].value<int64_t>();

    if (!max_period)             { SPDLOG_ERROR("Missing or invalid 'maximum_period'.");           return false; }
    if (!target_frames)          { SPDLOG_ERROR("Missing or invalid 'target_frame_nb'.");          return false; }
    if (!dataset_capture_mode_val){ SPDLOG_ERROR("Missing or invalid 'dataset_capture_mode'.");    return false; }
    if (!imu_collection_mode_val){ SPDLOG_ERROR("Missing or invalid 'imu_collection_mode'.");      return false; }
    if (!image_capture_rate_val) { SPDLOG_ERROR("Missing or invalid 'image_capture_rate'.");       return false; }
    if (!imu_sample_rate_hz_val) { SPDLOG_ERROR("Missing or invalid 'imu_sample_rate_hz'.");       return false; }
    if (!target_processing_stage_val){ SPDLOG_ERROR("Missing or invalid 'target_processing_stage'."); return false; }
    if (!capture_start_time_val) { SPDLOG_ERROR("Missing or invalid 'capture_start_time'.");       return false; }

    return validateRawParams(*max_period, *target_frames, *dataset_capture_mode_val,
                             *imu_collection_mode_val, *image_capture_rate_val,
                             static_cast<float>(*imu_sample_rate_hz_val),
                             *target_processing_stage_val);
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
    
    if (nb_frames == 0)
    {
        SPDLOG_ERROR("Target frame number must be at least 1");
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
                uint64_t capture_start_time)
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
    if (!candidate_folder.empty() && candidate_folder.back() != '/')
    {
        candidate_folder += '/';
    }
    folder_path = candidate_folder;  
    imu_log_file_path = folder_path + "imu_data.csv";
    
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
    target_frame_nb = static_cast<uint8_t>(*(config["target_frame_nb"].value<uint64_t>()));
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
        imu_sample_rate_hz      = other.imu_sample_rate_hz;
        target_processing_stage = other.target_processing_stage;
        stored_frame_ids        = other.stored_frame_ids;
    }
    return *this;
}

void Dataset::AddStoredFrameID(const std::tuple<uint8_t, uint64_t>& frame_id)
{
    if (std::find(stored_frame_ids.begin(), stored_frame_ids.end(), frame_id) == stored_frame_ids.end())
    {
        stored_frame_ids.push_back(frame_id);
    }
}

void Dataset::AddStoredFrameIDs(const std::vector<std::tuple<uint8_t, uint64_t>>& frame_ids)
{
    for (const auto& frame_id : frame_ids)
    {
        AddStoredFrameID(frame_id);
    }
}


bool Dataset::CreateConfigurationFile()
{
    auto tbl = toml::table{
        {"capture_start_time", static_cast<int64_t>(capture_start_time)}, 
        {"maximum_period", maximum_period},
        {"target_frame_nb", target_frame_nb},
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
        Json frame_metadata = DH::LoadFrameMetadataFromDisk(std::get<1>(frame_id), std::get<0>(frame_id), folder_path);
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

// Returns the value at j[key] as std::optional<T>, or nullopt if the key is
// absent, the value is null, or the JSON type does not match T.
template<typename T>
static std::optional<T> json_get(const Json& j, const std::string& key)
{
    if (!j.contains(key) || j.at(key).is_null())
        return std::nullopt;
    const auto& v = j.at(key);
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
        if (!v.is_number()) return std::nullopt;
    } else if constexpr (std::is_unsigned_v<T>) {
        // Accept both signed and unsigned JSON integers (enums and uint8_t are
        // stored as signed by nlohmann's default serializer), but reject negatives.
        if (!v.is_number_integer()) return std::nullopt;
        if (!v.is_number_unsigned() && v.get<int64_t>() < 0) return std::nullopt;
    } else if constexpr (std::is_integral_v<T>) {
        if (!v.is_number_integer()) return std::nullopt;
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (!v.is_string()) return std::nullopt;
    }
    return v.get<T>();
}

bool Dataset::fromJson(const Json& j)
{
    // Read every field through the type-safe helper; any wrong type or missing
    // field produces nullopt and is reported before we touch member variables.
    const auto folder_path_opt      = json_get<std::string>(j, "folder_path");
    const auto imu_log_path_opt     = json_get<std::string>(j, "imu_log_file_path");
    const auto capture_start_opt    = json_get<uint64_t>(j, "capture_start_time");
    const auto max_period_opt       = json_get<double>(j, "maximum_period");
    const auto target_frame_opt     = json_get<uint64_t>(j, "target_frame_nb");
    const auto capture_mode_opt     = json_get<uint64_t>(j, "dataset_capture_mode");
    const auto imu_mode_opt         = json_get<uint64_t>(j, "imu_collection_mode");
    const auto image_rate_opt       = json_get<uint64_t>(j, "image_capture_rate");
    const auto imu_rate_opt         = json_get<float>(j, "imu_sample_rate_hz");
    const auto proc_stage_opt       = json_get<uint64_t>(j, "target_processing_stage");

    if (!folder_path_opt)   { SPDLOG_ERROR("fromJson: missing or wrong-typed field 'folder_path'");           return false; }
    if (!imu_log_path_opt)  { SPDLOG_ERROR("fromJson: missing or wrong-typed field 'imu_log_file_path'");     return false; }
    if (!capture_start_opt) { SPDLOG_ERROR("fromJson: missing or wrong-typed field 'capture_start_time'");    return false; }
    if (!max_period_opt)    { SPDLOG_ERROR("fromJson: missing or wrong-typed field 'maximum_period'");        return false; }
    if (!target_frame_opt)  { SPDLOG_ERROR("fromJson: missing or wrong-typed field 'target_frame_nb'");       return false; }
    if (!capture_mode_opt)  { SPDLOG_ERROR("fromJson: missing or wrong-typed field 'dataset_capture_mode'");  return false; }
    if (!imu_mode_opt)      { SPDLOG_ERROR("fromJson: missing or wrong-typed field 'imu_collection_mode'");   return false; }
    if (!image_rate_opt)    { SPDLOG_ERROR("fromJson: missing or wrong-typed field 'image_capture_rate'");    return false; }
    if (!imu_rate_opt)      { SPDLOG_ERROR("fromJson: missing or wrong-typed field 'imu_sample_rate_hz'");    return false; }
    if (!proc_stage_opt)    { SPDLOG_ERROR("fromJson: missing or wrong-typed field 'target_processing_stage'"); return false; }

    if (!j.contains("frame_id_list") || !j.at("frame_id_list").is_array()) {
        SPDLOG_ERROR("fromJson: missing or wrong-typed field 'frame_id_list'");
        return false;
    }

    if (!validateRawParams(*max_period_opt, *target_frame_opt, *capture_mode_opt,
                           *imu_mode_opt, *image_rate_opt, *imu_rate_opt, *proc_stage_opt))
        return false;

    folder_path             = *folder_path_opt;
    imu_log_file_path       = *imu_log_path_opt;
    capture_start_time      = *capture_start_opt;
    maximum_period          = *max_period_opt;
    target_frame_nb         = static_cast<uint8_t>(*target_frame_opt);
    dataset_capture_mode    = static_cast<CAPTURE_MODE>(*capture_mode_opt);
    imu_collection_mode     = static_cast<IMU_COLLECTION_MODE>(*imu_mode_opt);
    image_capture_rate      = static_cast<uint8_t>(*image_rate_opt);
    imu_sample_rate_hz      = *imu_rate_opt;
    target_processing_stage = static_cast<ProcessingStage>(*proc_stage_opt);
    stored_frame_ids        = j.at("frame_id_list").get<std::vector<std::tuple<uint8_t, uint64_t>>>();

    return true;
}


bool Dataset::OverlapsWith(const Dataset& other) const
{
    uint64_t a_start = capture_start_time;
    uint64_t a_end   = capture_start_time + static_cast<uint64_t>(maximum_period * 1000);
    uint64_t b_start = other.capture_start_time;
    uint64_t b_end   = other.capture_start_time + static_cast<uint64_t>(other.maximum_period * 1000);

    return (a_start >= b_start && a_start <= b_end) ||   // A's start falls inside B
           (a_end   >= b_start && a_end   <= b_end) ||   // A's end falls inside B
           (a_start <= b_start && a_end   >= b_end);     // A fully contains B
}
