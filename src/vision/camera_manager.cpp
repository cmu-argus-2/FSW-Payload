#include "vision/camera_manager.hpp"
#include "core/data_handling.hpp"

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
    for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        if (cameras[i].GetStatus() == CAM_STATUS::ACTIVE && cameras[i].IsNewFrameAvailable())
        {
            Frame buffer_frame = cameras[i].GetBufferFrame();

            if (buffer_frame.GetProcessingStage() == ProcessingStage::NotPrefiltered)
            {
                buffer_frame.RunPrefiltering();
            }
            
            if (only_earth && buffer_frame.GetImageState() < ImageState::Earth)
            {
                SPDLOG_INFO("CAM{}: Frame skipped (not Earth)", cameras[i].GetID());
                cameras[i].SetOffNewFrameFlag();
                continue;
            }
            [[maybe_unused]] std::string img_path = DH::StoreFrameToDisk(buffer_frame, IMAGES_FOLDER);
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
                std::unique_lock<std::mutex> lock(capture_mode_mutex);
                capture_mode_cv.wait(lock, [this] { return !loop_flag.load() || !(capture_mode.load() != CAPTURE_MODE::IDLE); });
                
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

void CameraManager::StopLoops()
{
    display_flag.store(false);
    loop_flag.store(false);
    capture_mode_cv.notify_all();

    for (auto& camera : cameras) 
    {
        camera.StopCaptureLoop();
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
}


void CameraManager::SendCaptureRequest()
{
    SetCaptureMode(CAPTURE_MODE::CAPTURE_SINGLE);
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
                camera.Enable();
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