#ifndef DATASET_HPP
#define DATASET_HPP

#include <vision/camera_manager.hpp>
#include <imu/imu_manager.hpp>
#include <vision/frame.hpp>

#include <vector>
#include <string>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <unordered_map>

// Move all that to contexpr
#define DATASET_CONFIG_FILE_NAME "dataset_config.toml"
#define MAX_SAMPLES 255
#define DEFAULT_COLLECTION_PERIOD 600
#define ABSOLUTE_MINIMUM_PERIOD 0.1
#define ABSOLUTE_MAXIMUM_PERIOD 10800 // 3h

// Error codes TODO with framework

inline bool IsValidCaptureMode(CAPTURE_MODE value)
{
    return (value >= CAPTURE_MODE::PERIODIC && 
            value <= CAPTURE_MODE::PERIODIC_LDMK);
}

class Dataset
{

public:
    // Static methos
    static std::vector<std::string> ListAllStoredDatasets();
    static bool isValidConfigurationFile(const std::string& config_file_path);
    static bool isValidConfiguration(double max_period, uint16_t nb_frames, CAPTURE_MODE capture_mode, IMU_COLLECTION_MODE imu_collection_mode,
                                    uint8_t image_capture_rate, float imu_sample_rate_hz, ProcessingStage target_processing_stage,
                                    uint64_t capture_start_time);

    // Getters
    uint64_t GetCaptureStartTime() const { return capture_start_time; }
    uint16_t GetTargetFrameNb() const { return target_frame_nb; }
    double GetMaximumPeriod() const { return maximum_period; }
    CAPTURE_MODE GetDatasetCaptureMode() const { return dataset_capture_mode; }
    IMU_COLLECTION_MODE GetIMUCollectionMode() const { return imu_collection_mode; }
    uint8_t GetImageCaptureRate() const { return image_capture_rate; }
    float GetIMUSampleRateHz() const { return imu_sample_rate_hz; }
    ProcessingStage GetTargetProcessingStage() const { return target_processing_stage; }
    std::string GetFolderPath() const { return folder_path; }
    std::string GetIMUFilePath() const { return imu_log_file_path; }
    std::vector<std::tuple<uint8_t, uint64_t>> GetStoredFrameIDs() const { return stored_frame_ids; }

    Json toJson() const;
    bool fromJson(const Json& j);

    bool OverlapsWith(const Dataset& other) const;

    // Actual constructors ~ not to be used
    Dataset(double max_period, uint16_t nb_frames, CAPTURE_MODE capture_mode, IMU_COLLECTION_MODE imu_collection_mode,
            uint8_t image_capture_rate, float imu_sample_rate_hz, ProcessingStage target_processing_stage,
            uint64_t capture_start_time);

    // TODO: Rethink if function below is truly needed, seems redundant with json
    Dataset(const std::string& folder_path);

    Dataset& operator=(const Dataset& other);

private:
    std::string folder_path;
    std::string imu_log_file_path;
    
    uint64_t capture_start_time; // unix in ms. For scheduling
    double maximum_period;
    uint16_t target_frame_nb;
    CAPTURE_MODE dataset_capture_mode;
    IMU_COLLECTION_MODE imu_collection_mode;
    uint8_t image_capture_rate; // [s]
    float imu_sample_rate_hz; // [hz]
    ProcessingStage target_processing_stage;

    std::vector<std::tuple<uint8_t, uint64_t>> stored_frame_ids; // for statistics, not intended to be used for loading frames (too heavy)

    bool CreateConfigurationFile();
    
};

#endif // DATASET_HPP
