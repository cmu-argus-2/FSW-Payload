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



std::shared_ptr<DatasetManager> DatasetManager::Create(double max_period, uint16_t target_frame_nb, CAPTURE_MODE capture_mode, uint64_t capture_start_time,
                                                        IMU_COLLECTION_MODE imu_collection_mode, uint8_t image_capture_rate, float imu_sample_rate_hz, 
                                                        ProcessingStage target_processing_stage, std::string ds_key, 
                                                        CameraManager& cam_manager=sys::cameraManager(), IMUManager& imu_manager=sys::imuManager())
{
    auto instance = std::make_shared<DatasetManager>(max_period, target_frame_nb, capture_mode, imu_collection_mode, 
                                                        image_capture_rate, imu_sample_rate_hz, target_processing_stage, 
                                                        capture_start_time, cam_manager, imu_manager);
    std::lock_guard<std::mutex> lock(datasets_mtx);
    if (ds_key == DEFAULT_DS_KEY)
    {
        ds_key = std::to_string(instance->created_at);
    }
    active_datasets[ds_key] = instance;
    return instance;
}

std::shared_ptr<DatasetManager> DatasetManager::Create(const std::string& folder_path, std::string ds_key = DEFAULT_DS_KEY)
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


std::shared_ptr<DatasetManager> DatasetManager::GetActiveDataset(const std::string& key = DEFAULT_DS_KEY)
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


DatasetManager::DatasetManager(double max_period, uint16_t nb_frames, 
                                CAPTURE_MODE capture_mode,
                                IMU_COLLECTION_MODE imu_collection_mode,
                                uint8_t image_capture_rate, float imu_sample_rate_hz, 
                                ProcessingStage target_processing_stage,
                                uint64_t capture_start_time=timing::GetCurrentTimeMs(), 
                                CameraManager& cam_manager=sys::cameraManager(), 
                                IMUManager& imu_manager=sys::imuManager())
:
capture_start_time(capture_start_time),
maximum_period(max_period),
target_frame_nb(nb_frames),
dataset_capture_mode(capture_mode),
progress(nb_frames),
cameraManager(cam_manager),
imuManager(imu_manager),
imu_collection_mode(imu_collection_mode),
image_capture_rate(image_capture_rate),
imu_sample_rate_hz(imu_sample_rate_hz),
target_processing_stage(target_processing_stage)
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
capture_start_time(timing::GetCurrentTimeMs()),
maximum_period(DEFAULT_COLLECTION_PERIOD), // default
target_frame_nb(MAX_SAMPLES), // default
dataset_capture_mode(CAPTURE_MODE::PERIODIC), // default
imu_collection_mode(IMU_COLLECTION_MODE::GYRO_MAG_TEMP),
image_capture_rate(60),
imu_sample_rate_hz(1.0f),
target_processing_stage(ProcessingStage::NotPrefiltered),
progress(target_frame_nb),
cameraManager(sys::cameraManager()),
imuManager(sys::imuManager())
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

        std::optional<double> max_period = config["maximum_period"].value<double>();
        if (!max_period)
        {
            throw std::invalid_argument("Missing or invalid 'maximum_period' in configuration.");
        }
        maximum_period = *max_period; // dereference the contained value
        if (maximum_period < ABSOLUTE_MINIMUM_PERIOD)
        {
            SPDLOG_ERROR("Maximum period {} is below the absolute minimum period {}", maximum_period, ABSOLUTE_MINIMUM_PERIOD);
            throw std::invalid_argument("Maximum period is below the absolute minimum period.");
        }

        if (maximum_period > ABSOLUTE_MAXIMUM_PERIOD)
        {
            SPDLOG_ERROR("Maximum period {} is above the absolute maximum period {}", maximum_period, ABSOLUTE_MAXIMUM_PERIOD);
            throw std::invalid_argument("Maximum period is above the absolute maximum period.");
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

        std::optional<uint64_t> dataset_capture_mode_val = config["dataset_capture_mode"].value<uint64_t>();
        if (!dataset_capture_mode_val)
        {
            throw std::invalid_argument("Missing or invalid 'dataset_capture_mode' in configuration.");
        }
        dataset_capture_mode = static_cast<CAPTURE_MODE>(*dataset_capture_mode_val);
        if (!IsValidCaptureMode(dataset_capture_mode))
        {
            SPDLOG_ERROR("Invalid dataset capture mode {}", static_cast<int>(dataset_capture_mode));
            throw std::invalid_argument("Invalid dataset capture mode.");
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
        {"maximum_period", maximum_period},
        {"target_frames", target_frame_nb},
        {"dataset_capture_mode", dataset_capture_mode}
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
    // Launch the collection thread- is this necessary?
    // this will be launched from a command, that is in a thread itseld
    CollectionLoop();
    // collection_thread = std::thread(&DatasetManager::CollectionLoop, this);

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
    return (progress.current_frames >= target_frame_nb) || (timing::GetCurrentTimeMs() - capture_start_time > maximum_period * 1000);
}

void DatasetManager::CollectionLoop()
{
    // To consider: No two datasets will collect at the same time. 
    // Will have to be careful to guarantee the CameraManager is correctly configured
    // at the right time

    // wait until 
    
    while (timing::GetCurrentTimeMs() < capture_start_time)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // configure the imu manager for the dataset collection
    imuManager.SetLogFile(folder_path + "imu_data.csv");
    imuManager.SetSampleRate(1.0); // TODO: make this configurable
    imuManager.StartCollection();

    // configure the camera manager for the dataset collection
    // TODO:Define folder to store images on
    cameraManager.SetStorageFolder(folder_path);
    cameraManager.SetPeriodicCaptureRate(uint8_t(image_capture_rate));
    cameraManager.SetPeriodicFramesToCapture(target_frame_nb);
    cameraManager.SetCaptureMode(dataset_capture_mode);

    int no_frame_counter = 0;
    while (loop_flag.load() && !CheckTermination())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (no_frame_counter < cameraManager.GetCapturedFramesCount())
        {
            no_frame_counter = cameraManager.GetCapturedFramesCount();
            progress.Update(no_frame_counter, 1.0); // TODO: compute actual hit
            // TODO: store the frame name information
        }
    }

    // terminate data collection
    cameraManager.SetCaptureMode(CAPTURE_MODE::IDLE);
    cameraManager.SetStorageFolder(IMAGES_FOLDER); // Reset to default folder
    imuManager.Suspend();

    loop_flag.store(false);
    /*
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
        if (!getCameraManager().CountActiveCameras())
        {
            // TODO: Error code 
            SPDLOG_WARN("No active cameras.");
            // TODO: might need to just continue and wait for the camera monitor to kick in
            continue;
        }

        // Copy the latest frames 
        uint8_t nb_copied_frames = getCameraManager().CopyFrames(buffer_frames, earth_flag);
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
                    [[maybe_unused]] std::string path = DH::StoreFrameToDisk(frame, folder_path);
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
    */
}