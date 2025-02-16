#include <ostream>
#include <stdexcept>
#include "toml.hpp"
#include "spdlog/spdlog.h"

#include "vision/dataset.hpp"
#include "core/data_handling.hpp"
#include "core/timing.hpp"
#include "payload.hpp"


DatasetProgress::DatasetProgress(uint16_t target_nb_frames)
:
hit_ratio(1.0),
_progress_calls(0.0),
completion(0.0),
current_frames(0),
target_frames(target_nb_frames)
{}

void DatasetProgress::Update(uint16_t nb_new_frames, double instant_hit_ratio)
{
    current_frames += nb_new_frames;
    
    // cumulative average
    hit_ratio = (instant_hit_ratio + _progress_calls * hit_ratio) / (_progress_calls + 1);
    _progress_calls++;

    completion = static_cast<double>(current_frames) / static_cast<double>(target_frames);
    SPDLOG_INFO("Current progress: {} / {}", current_frames, target_frames);
}

// Registry for the datasets
std::unordered_map<std::string, std::shared_ptr<DatasetManager>> DatasetManager::active_datasets;
std::mutex DatasetManager::datasets_mtx;



std::shared_ptr<DatasetManager> DatasetManager::Create(double min_period, uint16_t nb_frames, DatasetType type, std::string ds_key)
{
    auto instance = std::make_shared<DatasetManager>(min_period, nb_frames, type);
    std::lock_guard<std::mutex> lock(datasets_mtx);
    if (ds_key == DEFAULT_DS_KEY)
    {
        ds_key = std::to_string(instance->created_at);
    }
    active_datasets[ds_key] = instance;
    return instance;
}

std::shared_ptr<DatasetManager> DatasetManager::Create(const std::string& folder_path, std::string ds_key)
{
    auto instance = std::make_shared<DatasetManager>(folder_path);
    std::lock_guard<std::mutex> lock(datasets_mtx);
    if (ds_key == DEFAULT_DS_KEY)
    {
        ds_key = std::to_string(instance->created_at);
    }
    active_datasets[ds_key] = instance;
    return instance;
}

std::shared_ptr<DatasetManager> DatasetManager::GetActiveDataset(const std::string& key)
{
    std::lock_guard<std::mutex> lock(datasets_mtx);
    auto it = active_datasets.find(key);
    return (it != active_datasets.end()) ? it->second : nullptr;
    // TODO: must check it's actually running (collection in progress)
}

void DatasetManager::StopDataset(const std::string& key)
{
    std::lock_guard<std::mutex> lock(datasets_mtx);
    auto it = active_datasets.find(key);
    if (it != active_datasets.end())
    {
        it->second->StopCollection();
        active_datasets.erase(it);
    }
}

std::vector<std::string> DatasetManager::ListActiveDatasets()
{
    std::lock_guard<std::mutex> lock(datasets_mtx);
    std::vector<std::string> keys;
    for (const auto& entry : active_datasets)
    {
        keys.push_back(entry.first);
    }
    return keys;
}

std::vector<std::string> DatasetManager::ListAllStoredDatasets()
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



DatasetManager::DatasetManager(double min_period, uint16_t nb_frames, DatasetType type)
:
minimum_period(min_period),
target_frame_nb(nb_frames),
dataset_type(type),
progress(nb_frames)
{
    created_at = timing::GetCurrentTimeMs();
    folder_path = DATASETS_FOLDER + std::to_string(created_at) + "/";

    bool res = DH::MakeNewDirectory(folder_path);
    if (!res)
    {
        SPDLOG_ERROR("Failed to create {}", folder_path);
    }

    CreateConfigurationFile(); 
}

DatasetManager::DatasetManager(const std::string& folder_path)
:
minimum_period(DEFAULT_COLLECTION_PERIOD), // default
target_frame_nb(MAX_SAMPLES), // default
dataset_type(DatasetType::EARTH_ONLY), // default
progress(target_frame_nb)
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

        std::optional<double> min_period = config["minimum_period"].value<double>();
        if (!min_period)
        {
            throw std::invalid_argument("Missing or invalid 'minimum_period' in configuration.");
        }
        minimum_period = *min_period;
        if (minimum_period < ABSOLUTE_MINIMUM_PERIOD)
        {
            SPDLOG_ERROR("Minimum period {} is below the absolute minimum period {}", minimum_period, ABSOLUTE_MINIMUM_PERIOD);
            throw std::invalid_argument("Minimum period is below the absolute minimum period.");
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

        std::optional<uint64_t> dataset_type_val = config["dataset_type"].value<uint64_t>();
        if (!dataset_type_val)
        {
            throw std::invalid_argument("Missing or invalid 'dataset_type' in configuration.");
        }
        dataset_type = static_cast<DatasetType>(*dataset_type_val);
        if (IsValidDatasetType(dataset_type))
        {
            SPDLOG_ERROR("Invalid dataset type {}", static_cast<int>(dataset_type));
            throw std::invalid_argument("Invalid dataset type.");
        }
    }
    catch (const std::exception& e)
    {
        SPDLOG_ERROR("Failed to parse configuration file: {}", e.what());
        throw std::runtime_error("Failed to parse configuration file or incorrect configuration.");
    }

}



void DatasetManager::CreateConfigurationFile()
{
    auto tbl = toml::table{
        {"minimum_period", minimum_period},
        {"target_frames", target_frame_nb},
        {"dataset_type", dataset_type}
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


bool DatasetManager::IsCompleted()
{
    return CheckTermination();
}


bool DatasetManager::StartCollection()
{
    // Create folder if it doesn't exists
    if (!DH::fs::exists(folder_path))
    {
        if (!DH::MakeNewDirectory(folder_path))
        {
            SPDLOG_ERROR("Failed to create {}", folder_path);
            return false;
        }
    }

    loop_flag.store(true);
    // Launch the collection thread
    collection_thread = std::thread(&DatasetManager::CollectionLoop, this);

    return true;
}


void DatasetManager::StopCollection()
{
    loop_flag.store(false);
    loop_cv.notify_all();
    if (collection_thread.joinable())
    {
        collection_thread.join();
    }
}

bool DatasetManager::Running()
{
    return loop_flag.load();
}

DatasetProgress DatasetManager::QueryProgress() const
{
    return progress;
}

bool DatasetManager::CheckTermination()
{
    return (progress.current_frames >= target_frame_nb) || (progress.current_frames >= MAX_SAMPLES);
}

void DatasetManager::CollectionLoop()
{
    
    // Prepare earth flag
    bool earth_flag = false;
    if ((dataset_type == DatasetType::EARTH_ONLY) || (dataset_type == DatasetType::LANDMARKS))
    {
        earth_flag = true;
    }

    // Prepare buffer for the frames
    std::vector<Frame> buffer_frames;
    buffer_frames.reserve(NUM_CAMERAS); // max possible, should never happen given our camera configuration

    // Error handling
    int no_frame_counter = 0;

    // Timing stuff
    const auto interval = std::chrono::duration_cast<timing::Clock::duration>(std::chrono::duration<double>(minimum_period)); // casting set it to nanoseconds given the clock
    auto next_time = timing::Clock::now();


    // Assumes folder exists
    while (loop_flag.load() && !CheckTermination())
    {
        // Wait for stopping condition or timeout based on the predetermined period 
        {
            std::unique_lock<std::mutex> lock(loop_mtx);
            loop_cv.wait_until(lock, next_time, [this] { return !loop_flag.load(); });
        }

        // Check if any camera is active 
        if (!sys::cameraManager().CountActiveCameras())
        {
            // TODO: Error code 
            SPDLOG_WARN("No active cameras.");
            // TODO: might need to just continue and wait for the camera monitor to kick in
            continue;
        }

        // Copy the latest frames 
        uint8_t nb_copied_frames = sys::cameraManager().CopyFrames(buffer_frames, earth_flag);
        SPDLOG_WARN("Number of copied frames {}", nb_copied_frames);

        if (nb_copied_frames > 0)
        {
            no_frame_counter = 0;
            int saved_frames = 0;

            // ---- for LANDMARKS --> save the images AND their associated RC and landmark data (as CSV files for a start)
            if (dataset_type == DatasetType::LANDMARKS) 
            {
                // TODO: Neural Engine pass
                // discard frames that are not ROI here ~ hit ratio for statistics

                // Save the frame and all its associated data in the folder
            }
            else // ---- for ANY and EARTH_ONLY --> just save the images 
            {   
                for (auto frame : buffer_frames)
                {
                    DH::StoreFrameToDisk(frame, folder_path);
                    saved_frames++;
                    progress.Update(saved_frames);
                    SPDLOG_WARN("Number of saved frames {}", saved_frames);
                }
            }

            // reset
            saved_frames = 0;
            buffer_frames.clear();

            if (CheckTermination())
            {
                SPDLOG_INFO("Completed dataset collection!");
                break;
            }

            // pause till next period
            next_time += interval;
            std::this_thread::sleep_until(next_time);
        }
        else
        {
            no_frame_counter++;
        }

        // TODO: some logic handling if we never get a frame for too many runs

    }
    

    loop_flag.store(false);

}