#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <unordered_map>

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
#define ABSOLUTE_MINIMUM_PERIOD 0.1
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

enum class DatasetType
{
    ANY = 0,
    EARTH_ONLY = 1,
    LANDMARKS = 2,
};

inline bool IsValidDatasetType(DatasetType value)
{
    return (value >= DatasetType::ANY && 
            value <= DatasetType::LANDMARKS);
}

class DatasetManager
{

public:

    // Static methods

    // It is recommended to have the Create functions under a try-except ot catch instantiation failures
    static std::shared_ptr<DatasetManager> Create(double min_period, uint16_t nb_frames, DatasetType type, std::string ds_key = DEFAULT_DS_KEY); // fine to pass string by value/copy
    // If the folder path does not exist or does not contain a config file, it throws.
    static std::shared_ptr<DatasetManager> Create(const std::string& folder_path, std::string key = DEFAULT_DS_KEY);

    static std::shared_ptr<DatasetManager> GetActiveDataset(const std::string& key);
    static void StopDataset(const std::string& key);
    static std::vector<std::string> ListActiveDatasets();

    static std::vector<std::string> ListAllStoredDatasets();



    // Actual constructors
    DatasetManager(double min_period, uint16_t nb_frames, DatasetType type);
    // If the folder path does not exist or does not contain a config file, it throws.
    DatasetManager(const std::string& folder_path);


    bool IsCompleted();

    bool StartCollection();
    void StopCollection();
    bool Running();
    const DatasetProgress& QueryProgress() const;


private:

    uint64_t created_at;
    std::string folder_path;
    double minimum_period;
    uint16_t target_frame_nb;
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


    static std::unordered_map<std::string, std::shared_ptr<DatasetManager>> active_datasets;
    static std::mutex datasets_mtx;

    // DataFormatter

};


#endif // DATASET_HPP