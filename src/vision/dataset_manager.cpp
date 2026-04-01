#include <fstream>
#include <ostream>
#include <stdexcept>
#include "toml.hpp"
#include "spdlog/spdlog.h"

#include "vision/dataset_manager.hpp"
#include "vision/dataset.hpp"
#include "core/data_handling.hpp"
#include "core/timing.hpp"
#include "payload.hpp"


DatasetProgress::DatasetProgress(uint8_t target_nb_frames)
:
hit_ratio(1.0),
_progress_calls(0.0),
completion(0.0),
current_frames(0),
target_frames(target_nb_frames)
{}

void DatasetProgress::Update(uint8_t nb_new_frames, double instant_hit_ratio)
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


std::shared_ptr<DatasetManager> DatasetManager::Create(double max_period, uint8_t target_frame_nb, CAPTURE_MODE capture_mode, uint64_t capture_start_time,
                            IMU_COLLECTION_MODE imu_collection_mode, uint8_t image_capture_rate, float imu_sample_rate_hz,
                            ProcessingStage target_processing_stage, std::string ds_key = DEFAULT_DS_KEY,
                            CameraManager& cam_manager = sys::cameraManager(), IMUManager& imu_manager = sys::imuManager(),
                            InferenceManager& inference_manager = sys::inferenceManager())
{

    Dataset dataset = Dataset(max_period, target_frame_nb, capture_mode, imu_collection_mode,
                                image_capture_rate, imu_sample_rate_hz, target_processing_stage,
                                capture_start_time);

    // if dataset overlaps others, return false
    for (const auto& entry : active_datasets)
    {
        auto existing_ds = entry.second;
        if (dataset.OverlapsWith(existing_ds->current_dataset)) {
            SPDLOG_ERROR("Failed to create DatasetManager: overlapping with an active dataset (key: {})", entry.first);
            throw std::invalid_argument("Overlapping dataset.");
        }
    }

    auto instance = std::make_shared<DatasetManager>(dataset, cam_manager, imu_manager, inference_manager);
    std::lock_guard<std::mutex> lock(datasets_mtx);
    if (ds_key == DEFAULT_DS_KEY)
    {
        ds_key = std::to_string(instance->created_at);
    }
    active_datasets[ds_key] = instance;
    return instance;
}

std::shared_ptr<DatasetManager> DatasetManager::Create(const std::string& folder_path, std::string ds_key = DEFAULT_DS_KEY,
                            CameraManager& cam_manager = sys::cameraManager(), IMUManager& imu_manager = sys::imuManager(),
                            InferenceManager& inference_manager = sys::inferenceManager())
{
    auto instance = std::make_shared<DatasetManager>(folder_path, cam_manager, imu_manager, inference_manager);
    std::lock_guard<std::mutex> lock(datasets_mtx);
    if (ds_key == DEFAULT_DS_KEY)
    {
        ds_key = std::to_string(instance->created_at);
    }
    active_datasets[ds_key] = instance;
    return instance;
}


std::shared_ptr<DatasetManager> DatasetManager::GetActiveDatasetManager(const std::string& key = DEFAULT_DS_KEY)
{
    std::lock_guard<std::mutex> lock(datasets_mtx);
    auto it = active_datasets.find(key);
    return (it != active_datasets.end()) ? it->second : nullptr;
    // TODO: must check it's actually running (collection in progress)
}

void DatasetManager::StopDatasetManager(const std::string& key)
{
    std::lock_guard<std::mutex> lock(datasets_mtx);
    auto it = active_datasets.find(key);
    if (it != active_datasets.end())
    {
        it->second->StopCollection();
        active_datasets.erase(it);
    }
}

std::vector<std::string> DatasetManager::ListActiveDatasetManagers()
{
    std::lock_guard<std::mutex> lock(datasets_mtx);
    std::vector<std::string> keys;
    for (const auto& entry : active_datasets)
    {
        keys.push_back(entry.first);
    }
    return keys;
}

DatasetManager::DatasetManager(Dataset dataset,
                                CameraManager& cam_manager, IMUManager& imu_manager,
                                InferenceManager& inference_manager)
:
current_dataset(dataset),
progress(dataset.GetTargetFrameNb()),
cameraManager(cam_manager),
imuManager(imu_manager),
inferenceManager(inference_manager),
created_at(timing::GetCurrentTimeMs())
{
}

DatasetManager::DatasetManager(const std::string& folder_path,
                                CameraManager& cam_manager, IMUManager& imu_manager,
                                InferenceManager& inference_manager)
:
current_dataset(folder_path),
progress(current_dataset.GetTargetFrameNb()),
cameraManager(cam_manager),
imuManager(imu_manager),
inferenceManager(inference_manager),
created_at(timing::GetCurrentTimeMs())
{
}

DatasetManager::~DatasetManager()
{
    if (Running())
    {
        StopCollection();
    }
    std::lock_guard<std::mutex> lock(datasets_mtx);
    for (auto it = active_datasets.begin(); it != active_datasets.end(); )
    {
        if (it->second.get() == this)
        {
            it = active_datasets.erase(it);
        }
        else
        {
            ++it;
        }
    }
}


bool DatasetManager::IsCompleted()
{
    return CheckTermination();
}


bool DatasetManager::StartCollection()
{
    if (loop_flag.load())
    {
        SPDLOG_WARN("Dataset collection already running");
        return false;
    }

    // Create folder if it doesn't exists
    if (!DH::fs::exists(current_dataset.GetFolderPath()))
    {
        if (!DH::MakeNewDirectory(current_dataset.GetFolderPath()))
        {
            SPDLOG_ERROR("Failed to create {}", current_dataset.GetFolderPath());
            return false;
        }
    }

    loop_flag.store(true);
    CollectionLoop();

    return true;
}


void DatasetManager::StopCollection()
{
    loop_flag.store(false);
    loop_cv.notify_all();

    cameraManager.SetCaptureMode(CAPTURE_MODE::IDLE);
    cameraManager.SetStorageFolder(IMAGES_FOLDER);
    imuManager.Suspend();
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
    return (progress.current_frames >= current_dataset.GetTargetFrameNb()) || (timing::GetCurrentTimeMs() - current_dataset.GetCaptureStartTime() > current_dataset.GetMaximumPeriod() * 1000);
}

void DatasetManager::ProcessFrames(
    const std::vector<std::tuple<uint8_t, uint64_t>>& frame_ids,
    std::vector<std::tuple<uint8_t, uint64_t>>& processed_frame_ids)
{
    const ProcessingStage target = current_dataset.GetTargetProcessingStage();
    const ProcessingStage capture_stage = CaptureModeToProcessingStage(current_dataset.GetDatasetCaptureMode());

    if (target <= capture_stage)
        return;

    for (const auto& frame_id : frame_ids)
    {
        if (std::find(processed_frame_ids.begin(), processed_frame_ids.end(), frame_id)
            != processed_frame_ids.end())
            continue;

        uint8_t cam_id = std::get<0>(frame_id);
        uint64_t timestamp = std::get<1>(frame_id);

        std::string img_path = current_dataset.GetFolderPath()
                             + "raw_" + std::to_string(timestamp)
                             + "_" + std::to_string(cam_id) + ".png";

        Frame frame;
        if (!DH::ReadImageFromDisk(img_path, frame, cam_id, timestamp))
        {
            SPDLOG_ERROR("Failed to load frame ({}, {}) for processing", cam_id, timestamp);
            processed_frame_ids.push_back(frame_id); // don't retry unloadable frames
            continue;
        }

        // Step 1: prefiltering — only if the capture mode didn't already filter earth-facing
        if (capture_stage < ProcessingStage::Prefiltered)
        {
            frame.RunPrefiltering();
            if (frame.GetImageState() < ImageState::Earth)
            {
                SPDLOG_INFO("Frame ({}, {}) rejected by prefiltering", cam_id, timestamp);
                processed_frame_ids.push_back(frame_id);
                continue;
            }
            if (target == ProcessingStage::Prefiltered)
            {
                DH::StoreFrameMetadataToDisk(frame, current_dataset.GetFolderPath());
                processed_frame_ids.push_back(frame_id);
                SPDLOG_INFO("Prefiltering complete for frame ({}, {})", cam_id, timestamp);
                continue;
            }
        }

        // Step 2: inference
        auto frame_ptr = std::make_shared<Frame>(frame);

        if (capture_stage >= ProcessingStage::RCNeted)
        {
            // RC was already done by the camera loop — restore regions from disk so
            // the frame's stage and region list are correctly populated for LD inference.
            Json metadata = DH::LoadFrameMetadataFromDisk(timestamp, cam_id, current_dataset.GetFolderPath());
            frame_ptr->fromJson(metadata);
        }

        EC status = inferenceManager.ProcessFrame(frame_ptr, target);
        if (status != EC::OK)
        {
            SPDLOG_ERROR("Inference failed for frame ({}, {}): error {}", cam_id, timestamp, to_uint8(status));
            processed_frame_ids.push_back(frame_id);
            continue;
        }

        const ImageState min_state = (target == ProcessingStage::RCNeted)
                                         ? ImageState::HasRegion
                                         : ImageState::HasLandmark;
        if (frame_ptr->GetImageState() < min_state)
        {
            SPDLOG_INFO("Frame ({}, {}) did not reach required image state for stage {}",
                        cam_id, timestamp, ProcessingStageToString(target));
            processed_frame_ids.push_back(frame_id);
            continue;
        }

        DH::StoreFrameMetadataToDisk(*frame_ptr, current_dataset.GetFolderPath());
        processed_frame_ids.push_back(frame_id);
        SPDLOG_INFO("Processing complete for frame ({}, {})", cam_id, timestamp);
    }
}

void DatasetManager::CollectionLoop()
{
    // TODO: No two datasets should collect at the same time. 
    // But multiple dataset managers can be created and running at the same time
    // Will have to be careful to guarantee the CameraManager is correctly configured
    // at the right time

    // wait until 
    
    while (timing::GetCurrentTimeMs() < current_dataset.GetCaptureStartTime())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // configure the camera manager for the dataset collection
    if (!cameraManager.PrepareForCapture())
    {
        loop_flag.store(false);
        throw std::runtime_error("Failed to enable cameras for dataset collection.");
    }

    cameraManager.SetStorageFolder(current_dataset.GetFolderPath());
    cameraManager.SetTargetProcessingStage(current_dataset.GetTargetProcessingStage());
    cameraManager.SetPeriodicCaptureRate(current_dataset.GetImageCaptureRate());
    cameraManager.SetPeriodicFramesToCapture(current_dataset.GetTargetFrameNb());
    cameraManager.SetCaptureMode(current_dataset.GetDatasetCaptureMode());

    // configure the imu manager for the dataset collection
    imuManager.SetLogFile(current_dataset.GetIMUFilePath());
    imuManager.SetSampleRate(current_dataset.GetIMUSampleRateHz());
    imuManager.SetCollectionMode(current_dataset.GetIMUCollectionMode());
    imuManager.StartCollection();

    int no_frame_counter = 0;
    std::vector<std::tuple<uint8_t, uint64_t>> processed_frame_ids;
    while (loop_flag.load() && !CheckTermination())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const int captured_frames = cameraManager.GetCapturedFramesCount();
        if (no_frame_counter < captured_frames)
        {
            const uint8_t new_frames = static_cast<uint8_t>(captured_frames - no_frame_counter);
            no_frame_counter = captured_frames;
            std::vector<std::tuple<uint8_t, uint64_t>> frame_ids = cameraManager.GetBufferFrameIDs();
            current_dataset.AddStoredFrameIDs(frame_ids);
            progress.Update(new_frames, 1.0); // TODO: compute actual hit

            ProcessFrames(frame_ids, processed_frame_ids);

        }
    }
    
    // Store dataset summary JSON inside the collection folder
    DH::StoreDatasetToDisk(current_dataset);

    // terminate data collection
    StopCollection();
}
