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

DatasetManager::DatasetManager(double min_period, uint32_t nb_frames, DatasetType type)
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

    // Then read config file and fill all parameters
    // TODO
    
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
    while (loop_flag.load() && CheckTermination())
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
                }
            }

            // reset
            saved_frames = 0;

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