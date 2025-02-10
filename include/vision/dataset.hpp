#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <cstdint>
#include <atomic>
#include <thread>
#include "core/timing.hpp"


/*
TODO
- Manages folder management for a single dataset process 
- Store frames, handles naming with DH, performs neural engine calls as necessary 
- Interface to query progress 
- Able to pause, resume, retrieve a process even after reboot 
- Is used by bioth OD and commands
- contains static methods to nalyze the other datasets 
*/

#define MAX_SAMPLES 1000
#define TIMEOUT_NO_DATA 500 
#define DATASET_CONFIG_NAME "config.toml"

// Error codes TODO with framework

struct DatasetProgress
{
    
    float hit_ratio; // ROI_IMG / TOTAL_IMG
    int current_frames;
    uint8_t completion; // as a %
    
    DatasetProgress();
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
    DatasetManager(const std::string& folder_path);

    bool IsCompleted();

    void StartCollection();
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
    void CollectionLoop();
    std::atomic<bool> loop_flag = false;
    std::thread collection_thread;

    void CreateConfigurationFile();
    


    // DataFormatter

};


#endif // DATASET_HPP