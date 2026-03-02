#ifndef DATASET_MANAGER_HPP
#define DATASET_MANAGER_HPP

#include <vision/dataset.hpp>
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
#define DATASET_CONFIG_FILE_NAME "config.toml"
#define MAX_SAMPLES 1000
#define TIMEOUT_NO_DATA 500 
#define DEFAULT_COLLECTION_PERIOD 600
#define ABSOLUTE_MINIMUM_PERIOD 0.1
#define ABSOLUTE_MAXIMUM_PERIOD 10800 // 3h
#define DEFAULT_DS_KEY "None"

// Error codes TODO with framework

struct DatasetProgress
{
    
    double hit_ratio; // ROI_IMG / TOTAL_IMG - Mostly for statistics
    double _progress_calls;
    double completion; // as a %
    uint16_t current_frames;
    const uint16_t target_frames;

    DatasetProgress(uint16_t target_nb_frames);
    void Update(uint16_t nb_new_frames, double instant_hit_ratio = 1.0f);
};


class DatasetManager
{

public:

    // Static methods

    // It is recommended to have the Create functions under a try-except to catch instantiation failures
    static std::shared_ptr<DatasetManager> Create(double max_period, uint16_t target_frame_nb, CAPTURE_MODE capture_mode, uint64_t capture_start_time,
                                                  IMU_COLLECTION_MODE imu_collection_mode, uint8_t image_capture_rate, float imu_sample_rate_hz, 
                                                  ProcessingStage target_processing_stage, std::string ds_key, CameraManager& cam_manager, IMUManager& imu_manager);
    // If the folder path does not exist or does not contain a config file, it throws.
    static std::shared_ptr<DatasetManager> Create(const std::string& folder_path, std::string key);

    static std::shared_ptr<DatasetManager> GetActiveDatasetManager(const std::string& key);
    static void StopDatasetManager(const std::string& key);
    static std::vector<std::string> ListActiveDatasetManagers();

    bool IsCompleted();

    bool StartCollection();
    void StopCollection();
    bool Running();
    // Copy is easier (and cheap here), instead of dealing with all the multithreading
    DatasetProgress QueryProgress() const;

    CameraManager& getCameraManager() { return cameraManager; }
    IMUManager& getIMUManager() { return imuManager; }

    uint64_t GetCaptureStartTime() const { return current_dataset.GetCaptureStartTime(); }

    // Actual constructors ~ not to be used
    DatasetManager(Dataset dataset, CameraManager& cam_manager, IMUManager& imu_manager);
    // If the folder path does not exist or does not contain a config file, it throws.
    DatasetManager(const std::string& folder_path);


private:

    uint64_t created_at;

    Dataset current_dataset;

    // will this be an issue in terms of memory efficiency?
    CameraManager& cameraManager;
    IMUManager& imuManager;

    DatasetProgress progress;

    bool CheckTermination();
    // Runs the loop that periodically takes the latest frames, performs any necessary preprocessing, and stores the data in the corresponding folder
    void CollectionLoop();
    std::atomic<bool> loop_flag = false;
    
    std::mutex loop_mtx;
    std::condition_variable loop_cv;


    static std::unordered_map<std::string, std::shared_ptr<DatasetManager>> active_datasets;
    static std::mutex datasets_mtx;

    // DataFormatter

};

#endif // DATASET_MANAGER_HPP