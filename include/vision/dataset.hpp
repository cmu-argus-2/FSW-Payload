#ifndef DATASET_HPP
#define DATASET_HPP


#include <vision/camera_manager.hpp>
#include <imu/imu_manager.hpp>
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
#define DEFAULT_COLLECTION_PERIOD 10
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

inline bool IsValidCaptureMode(CAPTURE_MODE value)
{
    return (value >= CAPTURE_MODE::PERIODIC && 
            value <= CAPTURE_MODE::PERIODIC_LDMK);
}

class DatasetManager
{

public:

    // Static methods

    // It is recommended to have the Create functions under a try-except to catch instantiation failures
    static std::shared_ptr<DatasetManager> Create(double min_period, uint16_t nb_frames, CAPTURE_MODE capture_mode, std::string ds_key, 
        CameraManager& cam_manager, IMUManager& imu_manager); // fine to pass string by value/copy
    // If the folder path does not exist or does not contain a config file, it throws.
    static std::shared_ptr<DatasetManager> Create(const std::string& folder_path, std::string key);

    static std::shared_ptr<DatasetManager> GetActiveDataset(const std::string& key);
    static void StopDataset(const std::string& key);
    static std::vector<std::string> ListActiveDatasets();

    static std::vector<std::string> ListAllStoredDatasets();

    bool IsCompleted();

    bool StartCollection();
    void StopCollection();
    bool Running();
    // Copy is easier (and cheap here), instead of dealing with all the multithreading
    DatasetProgress QueryProgress() const;

    CameraManager& getCameraManager() { return cameraManager; }
    IMUManager& getIMUManager() { return imuManager; }


    // Actual constructors ~ not to be used
    DatasetManager(double max_period, uint16_t nb_frames, CAPTURE_MODE capture_mode, 
                    CameraManager& cam_manager, IMUManager& imu_manager);
    // If the folder path does not exist or does not contain a config file, it throws.
    DatasetManager(const std::string& folder_path);



private:

    uint64_t created_at;
    std::string folder_path;
    
    // uint64_t capture_start_time; // unix in ms. For scheduling
    double maximum_period;
    uint16_t target_frame_nb;
    CAPTURE_MODE dataset_capture_mode; // TODO: can't be idle or capture single
    // uint8_t imu capture mode; // none, gyro only, gyro + temp, gyro + temp + mag
    // capture rate camera and imu
    // target processing of camera data before saving (e.g. prefiltering, compression...)

    // will this be an issue in terms of memory efficiency?
    CameraManager& cameraManager;
    IMUManager& imuManager;

    DatasetProgress progress;

    bool CheckTermination();
    // Runs the loop that periodically takes the latest frames, performs any necessary preprocessing, and stores the data in the corresponding folder
    void CollectionLoop();
    std::atomic<bool> loop_flag = false;
    std::thread collection_thread;

    void CreateConfigurationFile();
    
    std::mutex loop_mtx;
    std::condition_variable loop_cv;


    static std::unordered_map<std::string, std::shared_ptr<DatasetManager>> active_datasets;
    static std::mutex datasets_mtx;

    // DataFormatter

};

#endif // DATASET_HPP