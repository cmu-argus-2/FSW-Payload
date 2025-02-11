#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>


/*
TODO
- Manages folder management for a single dataset process 
- Store frames, handles naming with DH, performs neural engine calls as necessary 
- Interface to query progress 
- Able to pause, resume, retrieve a process even after reboot 
- Is used by bioth OD and commands
- contains static methods to nalyze the other datasets 
*/

#define DATASET_CONFIG_FILE_NAME "config.toml"
#define MAX_SAMPLES 1000
#define TIMEOUT_NO_DATA 500 
#define DEFAULT_COLLECTION_PERIOD 10

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

enum class DatasetType 
{
    ANY,
    EARTH_ONLY,
    LANDMARKS
};


class DatasetManager
{

public:

    DatasetManager(double min_period, uint32_t nb_frames, DatasetType type);
    // If the folder path does not exist or does not contain a config file, it throws.
    DatasetManager(const std::string& folder_path);

    bool IsConfigured();
    bool IsCompleted();

    bool StartCollection();
    void StopCollection();
    

    const DatasetProgress& QueryProgress() const;

    static std::vector<std::string> ListDatasets();
    



private:

    uint64_t created_at;
    std::string folder_path;
    double minimum_period;
    uint32_t target_frame_nb;
    DatasetType dataset_type;

    DatasetProgress progress;

    bool CheckTermination();
    // Runs the loop that periodically takes the latest frames, performs any necessary preprocessing, and stores the data in the corresponding folder
    void CollectionLoop();
    std::atomic<bool> loop_flag = false;
    std::thread collection_thread;

    void CreateConfigurationFile();
    
    std::mutex loop_mtx;
    std::condition_variable loop_cv;

    // DataFormatter

};


#endif // DATASET_HPP