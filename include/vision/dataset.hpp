#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <cstdint>
#include "core/data_handling.hpp"
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

#define TIMEOUT_NO_DATA 500 

// Error codes TODO with framework

struct DatasetProgress
{
    uint8_t completion; // as a %
    int current_frames;
    float hit_ratio; // ROI_IMG / TOTAL_IMG

    DatasetProgress();
};



class DatasetManager
{

public:

    DatasetManager();

    bool LoadLatestDatasetConfig();
    bool LoadDataset(const std::string& folder_path);
    bool Reconfigure(); // change dynamically
    bool IsCompleted() const;

    void StartCollection();
    void StopCollection();
    

    DatasetProgress QueryProgress() const;

    static std::vector<std::string> ListDatasets();
    



private:

    uint64_t created_at;
    std::string folder_name;
    int nb_frames;
    int period;
    bool earth_filter;
    bool ld_flag;

    // DataFormatter

};


#endif // DATASET_HPP