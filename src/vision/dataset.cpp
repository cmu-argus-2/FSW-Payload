#include "vision/dataset.hpp"
#include "core/data_handling.hpp"
#include "spdlog/spdlog.h"
#include "toml.hpp"
#include <ostream>


DatasetProgress::DatasetProgress()
:
hit_ratio(0.0f),
current_frames(0),
completion(0)
{}

DatasetManager::DatasetManager(double min_period, uint32_t nb_frames, DatasetType type)
:
minimum_period(min_period),
target_frame_nb(nb_frames),
dataset_type(type)
{
    created_at = timing::GetCurrentTimeMs();
    folder_path = DATASETS_FOLDER + std::to_string(created_at) + "/";

    bool res = DH::MakeNewDirectory(folder_path);
    if (res)
    {
        SPDLOG_INFO("Created new dataset folder at {}", folder_path);
    } 
    else
    {
        SPDLOG_ERROR("Failed to create {}", folder_path);
    }

    CreateConfigurationFile(); 
    
}

DatasetManager::DatasetManager(const std::string& folder_path)
{
    // check if path exists
    // Then read config file and fill all parameters
    
}

void DatasetManager::CreateConfigurationFile()
{

    auto tbl = toml::table{
        {"minimum_period", minimum_period},
        {"target_frames", target_frame_nb},
        {"dataset_type", dataset_type}
    };

    // Write to file
    std::ofstream file(folder_path + DATASET_CONFIG_NAME);
    if (file.is_open())
    {
        file << tbl;
        file.close();
        SPDLOG_INFO("Config file saved to: {}", folder_path + DATASET_CONFIG_NAME);
    }
    else
    {
        SPDLOG_ERROR("Failed to open file for writing");
    }
    // TODO: handle fail cases

}

bool DatasetManager::IsCompleted()
{
    return CheckTermination();
}


void DatasetManager::StartCollection()
{
    loop_flag.store(true);

    // Launch the collection thread
    collection_thread = std::thread(&DatasetManager::CollectionLoop, this);
}


void DatasetManager::StopCollection()
{
    loop_flag.store(false);
    if (collection_thread.joinable())
    {
        collection_thread.join();
    }
}


const DatasetProgress& DatasetManager::QueryProgress() const
{
    return progress;
}

bool DatasetManager::CheckTermination()
{
    return (progress.current_frames >= target_frame_nb) || (progress.current_frames <= MAX_SAMPLES);
}

void DatasetManager::CollectionLoop()
{


    while (loop_flag.load() && CheckTermination())
    {
        // TODO
    }


}