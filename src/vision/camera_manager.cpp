#include "vision/camera_manager.hpp"
#include "core/data_handling.hpp"
#include "inference/orchestrator.hpp"

CameraManager::CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs) 
:
capture_mode(CAPTURE_MODE::IDLE),
camera_configs(camera_configs),
cameras{{Camera(camera_configs[0].id, camera_configs[0].path), 
    Camera(camera_configs[1].id, camera_configs[1].path), 
    Camera(camera_configs[2].id, camera_configs[2].path), 
    Camera(camera_configs[3].id, camera_configs[3].path)}}
{
    _UpdateCamStatus();
    SPDLOG_INFO("Camera Manager initialized");
}

void CameraManager::_UpdateCamStatus()
{
    // Initialize cam_status based on the status of each camera
    for (size_t i = 0; i < NUM_CAMERAS; ++i) {
        cam_status[i] = cameras[i].GetStatus();
    }
}


void CameraManager::CaptureFrames()
{
    for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        if (cameras[i].GetStatus() == CAM_STATUS::ACTIVE)
        {   
            cameras[i].CaptureFrame();
        }
    }
}


uint8_t CameraManager::SaveLatestFrames(bool only_earth)
{
    uint8_t save_count = 0;
    const bool needs_prefilter = only_earth || GetTargetProcessingStage() >= ProcessingStage::Prefiltered;
    buffer_frame_ids.clear();
    for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        if (cameras[i].GetStatus() == CAM_STATUS::ACTIVE && cameras[i].IsNewFrameAvailable())
        {
            Frame buffer_frame = cameras[i].GetBufferFrame();

            if (needs_prefilter && buffer_frame.GetProcessingStage() == ProcessingStage::NotPrefiltered)
            {
                buffer_frame.RunPrefiltering();
            }
            
            if (only_earth && buffer_frame.GetImageState() < ImageState::Earth)
            {
                SPDLOG_INFO("CAM{}: Frame skipped (not Earth)", cameras[i].GetID());
                cameras[i].SetOffNewFrameFlag();
                continue;
            }
            [[maybe_unused]] std::string img_path = DH::StoreFrameToDisk(buffer_frame, GetStorageFolder());
            buffer_frame_ids.push_back(std::make_tuple(cameras[i].GetID(), buffer_frame.GetTimestamp()));
            cameras[i].SetOffNewFrameFlag();
            save_count++;
        }  
    }
    return save_count;
}


uint8_t CameraManager::CopyFrames(std::vector<Frame>& vec_frames, bool only_earth)
{

    uint8_t frame_count = 0;
    vec_frames.reserve(vec_frames.size() + NUM_CAMERAS); // assumes void vector

    for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        if (cameras[i].GetStatus() == CAM_STATUS::ACTIVE)
        {   
            Frame new_frame;
            cameras[i].CopyBufferFrame(new_frame);
            vec_frames.emplace_back(std::move(new_frame)); // Move to avoid copying
            frame_count++;
        }
    }

    return frame_count;
}



void CameraManager::RunLoop()
{
    loop_flag.store(true);

    auto last_health_check_time = std::chrono::high_resolution_clock::now(); // Track health check timing
    auto current_capture_time = std::chrono::high_resolution_clock::now();
    auto last_capture_time = std::chrono::high_resolution_clock::now();

    while (loop_flag.load()) 
    {

        // Check for incoming camera commands
        switch (capture_mode.load())
        {
            case CAPTURE_MODE::IDLE:
            {
                _AutoDisableIfNeeded();
                std::unique_lock<std::mutex> lock(capture_mode_mutex);
                capture_mode_cv.wait(lock, [this] {return !loop_flag.load() || capture_mode.load() != CAPTURE_MODE::IDLE;});
                
                break;
            }

            case CAPTURE_MODE::CAPTURE_SINGLE: // Response to a command
            {
                int single_frames_captured = SaveLatestFrames();
                if (single_frames_captured > 0)
                {
                    SPDLOG_INFO("Single capture request completed: {} frame(s) captured", single_frames_captured);
                }
                // TODO should be a way to ACK the command here, whether this is a success or failure
                SetCaptureMode(CAPTURE_MODE::IDLE);
                break;
            }

            case CAPTURE_MODE::PERIODIC:
            {
                current_capture_time = std::chrono::high_resolution_clock::now();
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_capture_time - last_capture_time).count();
                if (elapsed_seconds >= periodic_capture_rate) 
                {
                    // not an issue if we exceed a bit
                    bool only_earth = false;
                    periodic_frames_captured += SaveLatestFrames(only_earth);
                    SPDLOG_INFO("Periodic capture request: {}/{} frames captured", periodic_frames_captured, periodic_frames_to_capture);

                    if (periodic_frames_captured >= periodic_frames_to_capture)
                    {
                        SPDLOG_INFO("Periodic capture request completed");
                        SetCaptureMode(CAPTURE_MODE::IDLE);
                        periodic_frames_captured = 0;
                        periodic_frames_to_capture = DEFAULT_PERIODIC_FRAMES_TO_CAPTURE; // Reset to default
                        break;
                    }
                    last_capture_time = current_capture_time; // Update last capture time

                }
                break;
            }

            case CAPTURE_MODE::PERIODIC_EARTH:
            {
                current_capture_time = std::chrono::high_resolution_clock::now();
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_capture_time - last_capture_time).count();
                if (elapsed_seconds >= periodic_capture_rate) 
                {
                    // not an issue if we exceed a bit
                    bool only_earth = true;
                    periodic_frames_captured += SaveLatestFrames(only_earth);
                    SPDLOG_INFO("Periodic capture request: {}/{} frames captured", periodic_frames_captured, periodic_frames_to_capture);

                    if (periodic_frames_captured >= periodic_frames_to_capture)
                    {
                        SPDLOG_INFO("Periodic capture request completed");
                        SetCaptureMode(CAPTURE_MODE::IDLE);
                        periodic_frames_captured = 0;
                        periodic_frames_to_capture = DEFAULT_PERIODIC_FRAMES_TO_CAPTURE; // Reset to default
                        break;
                    }
                    last_capture_time = current_capture_time; // Update last capture time

                }
                break;
            }

            case CAPTURE_MODE::PERIODIC_ROI:
            {
                static Inference::Orchestrator orchestrator;
                const ProcessingStage requested_stage = GetTargetProcessingStage();

                current_capture_time = std::chrono::high_resolution_clock::now();
                auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(current_capture_time - last_capture_time).count();
                if (elapsed_seconds >= periodic_capture_rate)
                {
                    uint8_t roi_frames_captured = 0;
                    std::vector<Frame> captured_frames;

                    // capture and collect frames first
                    for (std::size_t i = 0; i < NUM_CAMERAS; ++i)
                    {
                        if (cameras[i].GetStatus() == CAM_STATUS::ACTIVE && cameras[i].IsNewFrameAvailable())
                        {
                            Frame buffer_frame = cameras[i].GetBufferFrame();

                            if (buffer_frame.GetProcessingStage() == ProcessingStage::NotPrefiltered)
                            {
                                buffer_frame.RunPrefiltering();
                            }

                            if (buffer_frame.GetImageState() < ImageState::Earth)
                            {
                                SPDLOG_INFO("CAM{}: Frame skipped (not Earth)", cameras[i].GetID());
                                cameras[i].SetOffNewFrameFlag();
                                continue;
                            }

                            DH::StoreFrameToDisk(buffer_frame, GetStorageFolder());
                            captured_frames.push_back(buffer_frame);
                            cameras[i].SetOffNewFrameFlag();
                        }
                    }

                    // TODO: figure out memory issues, i think disabling might help...
                    std::array<bool, NUM_CAMERAS> off_cameras;
                    DisableCameras(off_cameras);
                    SPDLOG_INFO("Cameras disabled before inference.");

                    for (auto& frame : captured_frames)
                    {
                        std::shared_ptr<Frame> frame_ptr = std::make_shared<Frame>(frame);
                        orchestrator.GrabNewImage(frame_ptr);
                        spdlog::info("Running ROI inference on camera {}", frame.GetCamID());
                        EC status;
                        if (requested_stage == ProcessingStage::RCNeted)
                        {
                            status = orchestrator.ExecRCInference();
                        }
                        else
                        {
                            status = orchestrator.ExecFullInference();
                        }
                        if (status == EC::OK)
                        {
                            DH::StoreFrameMetadataToDisk(*frame_ptr, GetStorageFolder());
                            spdlog::info("Frame metadata JSON saved for camera {}", frame.GetCamID());
                            roi_frames_captured++;
                        }
                        else
                        {
                            spdlog::error("ROI inference failed with error code: {}", to_uint8(status));
                        }
                    }

                    // need to reenable
                    std::array<bool, NUM_CAMERAS> on_cameras;
                    EnableCameras(on_cameras);
                    SPDLOG_INFO("Cameras re-enabled after inference.");

                    periodic_frames_captured += roi_frames_captured;
                    spdlog::info("Periodic ROI capture: {}/{} frames processed", periodic_frames_captured, periodic_frames_to_capture);

                    if (periodic_frames_captured >= periodic_frames_to_capture)
                    {
                        spdlog::info("Periodic ROI capture request completed");
                        SetCaptureMode(CAPTURE_MODE::IDLE);
                        periodic_frames_captured = 0;
                        periodic_frames_to_capture = DEFAULT_PERIODIC_FRAMES_TO_CAPTURE;
                        break;
                    }

                    last_capture_time = current_capture_time;
                }
                break;
            }

            default:
                SPDLOG_WARN("Unknown capture mode: {}", static_cast<uint8_t>(capture_mode.load()));
                break;
        }
        
        // Perform health check periodically
        auto current_health_check_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(current_health_check_time - last_health_check_time).count() >= CAMERA_HEALTH_CHECK_INTERVAL) 
        {
            _PerformCameraHealthCheck();
            last_health_check_time = current_health_check_time;
        }

        _UpdateCamStatus();

    }

    SPDLOG_INFO("Exiting Camera Manager Run Loop");
    SetDisplayFlag(false);
}



void CameraManager::RunDisplayLoop()
{

    int active_cams = 0;
    
    while (display_flag.load() && loop_flag.load()) 
    {
        active_cams = 0;
        
        for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
        {
            auto& cam = cameras[i]; // Alias for readability

            if (cam.GetStatus() == CAM_STATUS::ACTIVE) 
            {
                ++active_cams;

                if (cam.IsNewFrameAvailable())
                {
                    cam.DisplayLastFrame();
                }
            }
        }

        if (active_cams == 0) 
        {
            SPDLOG_WARN("No cameras are turned on. Exiting display loop.");
            break;
        }

        if (!display_flag.load() || !loop_flag.load()) 
        {
            SPDLOG_WARN("Display loop terminated by flags.");
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30)); // Just displaying
    }

    display_flag.store(false);
    cv::destroyAllWindows();

    SPDLOG_INFO("Exiting Camera Manager Display Loop");
}




CameraConfig* CameraManager::GetCameraConfig(int cam_id) 
{
    for (auto& config : camera_configs)
    {
        if (config.id == cam_id)
        {
            return &config; // Return a pointer to the found config
        }
    }
    return nullptr; // Return nullptr if the ID is not found
}

int CameraManager::GetCapturedFramesCount() const
{
    return periodic_frames_captured.load();
}

std::vector<std::tuple<uint8_t, uint64_t>> CameraManager::GetBufferFrameIDs() const
{
    return buffer_frame_ids;
}

void CameraManager::StopLoops()
{
    display_flag.store(false);
    loop_flag.store(false);
    capture_mode_cv.notify_all();
    auto_disable_after_capture.store(false);

    for (auto& camera : cameras) 
    {
        camera.Disable();
    }

    SPDLOG_INFO("Stopped camera loops...");
}

void CameraManager::SetDisplayFlag(bool display_flag)
{
    this->display_flag.store(display_flag);
}

bool CameraManager::GetDisplayFlag() const
{
    return display_flag.load();
}

void CameraManager::SetCaptureMode(CAPTURE_MODE mode)
{
    capture_mode.store(mode);
    capture_mode_cv.notify_all();
}


bool CameraManager::SendCaptureRequest()
{
    if (!PrepareForCapture())
    {
        return false;
    }
    SetCaptureMode(CAPTURE_MODE::CAPTURE_SINGLE);
    return true;
}


void CameraManager::SetPeriodicCaptureRate(uint8_t period)
{
    periodic_capture_rate = period;
    SPDLOG_INFO("Periodic capture rate set to: {} seconds", period);
}


void CameraManager::SetPeriodicFramesToCapture(uint8_t frames)
{
    periodic_frames_to_capture = frames;
    SPDLOG_INFO("Periodic frames to capture set to: {}", frames);
}


void CameraManager::SetStorageFolder(const std::string& s)
{
    if (!DH::fs::exists(s))
    {
        if (!DH::MakeNewDirectory(s))
        {
            SPDLOG_ERROR("Failed to create storage folder: {}", s);
            return;
        }
    }
    std::lock_guard<std::mutex> lock(storage_folder_m);
    storage_folder = s;
    SPDLOG_INFO("Camera storage folder set to: {}", s);
}

void CameraManager::SetTargetProcessingStage(ProcessingStage stage)
{
    target_processing_stage.store(stage);
}

std::string CameraManager::GetStorageFolder()
{
    std::lock_guard<std::mutex> lock(storage_folder_m);
    return storage_folder;
}

ProcessingStage CameraManager::GetTargetProcessingStage() const
{
    return target_processing_stage.load();
}

bool CameraManager::PrepareForCapture()
{
    const int active_before = CountActiveCameras();
    if (active_before == NUM_CAMERAS)
    {
        auto_disable_after_capture.store(false);
        return true;
    }

    std::array<bool, NUM_CAMERAS> on_cameras;
    int nb_enabled = EnableCameras(on_cameras);
    if (CountActiveCameras() == 0)
    {
        SPDLOG_ERROR("Failed to enable any cameras for capture.");
        auto_disable_after_capture.store(false);
        return false;
    }

    auto_disable_after_capture.store(active_before == 0);
    SPDLOG_INFO("Prepared {} additional camera(s) for capture.", nb_enabled);
    return true;
}

void CameraManager::_AutoDisableIfNeeded()
{
    if (!auto_disable_after_capture.exchange(false))
    {
        return;
    }

    std::array<bool, NUM_CAMERAS> off_cameras;
    int nb_disabled = DisableCameras(off_cameras);
    SPDLOG_INFO("Auto-disabled {} camera(s) after capture completion.", nb_disabled);
}


bool CameraManager::EnableCamera(int cam_id)
{
    for (auto& camera : cameras) 
    {
        if (camera.GetID() == cam_id)
        {
            return camera.Enable();
        }
    }
    return false;
}


bool CameraManager::DisableCamera(int cam_id)
{
    for (auto& camera : cameras) 
    {
        if (camera.GetID() == cam_id)
        {
            return camera.Disable();
        }
    }
    return false;
}


int CameraManager::EnableCameras(std::array<bool, NUM_CAMERAS>& id_activated_cams)
{
    id_activated_cams.fill(false); // Initialize to false
    int count = 0;

    for (size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        cameras[i].Enable(); 

        if (cameras[i].GetStatus() == CAM_STATUS::ACTIVE)
        {
            id_activated_cams[i] = true;
            count++;
        }
    }
    _UpdateCamStatus();
    return count;
}


int CameraManager::DisableCameras(std::array<bool, NUM_CAMERAS>& id_disabled_cams)
{
    id_disabled_cams.fill(false); // Initialize to false
    int count = 0;

    for (size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        cameras[i].Disable();

        if (cameras[i].GetStatus() == CAM_STATUS::INACTIVE)
        {
            id_disabled_cams[i] = true;
            count++;
        }
    }

    _UpdateCamStatus();
    return count;
}

int CameraManager::CountActiveCameras() const
{
    int active_count = 0;
    for (auto& camera : cameras) 
    {
        // access atomic variable
        if (camera.GetStatus() == CAM_STATUS::ACTIVE)
        {
            active_count++;
        }
    }
    return active_count;
}

CAPTURE_MODE CameraManager::GetCaptureMode() const
{
    return capture_mode.load(); // atomic
}   

void CameraManager::FillCameraStatus(uint8_t* status) 
{
    for (size_t i = 0; i < NUM_CAMERAS; ++i) {

        status[i] = static_cast<uint8_t>(this->cam_status[i]);

    }
}

void CameraManager::_PerformCameraHealthCheck()
{
    for (auto& camera : cameras) 
    {
        switch (camera.GetStatus())
        {
            case CAM_STATUS::ACTIVE:
            {
                if (camera.GetLastError() != CAM_ERROR::NO_ERROR) 
                {
                    // Restart the camera
                    camera.Disable();
                    camera.Enable();
                }
                break;
            }

            case CAM_STATUS::INACTIVE:
            {
                break;
            }

            default:
                // Restart to get out of an undefined state
                camera.Disable();
                camera.Enable();
                break;
        }
        
    }
}
