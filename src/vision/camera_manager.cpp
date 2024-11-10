#include "vision/camera_manager.hpp"
#include "payload.hpp"

CameraManager::CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs) 
:
capture_mode(CAPTURE_MODE::IDLE),
camera_configs(camera_configs),
cameras({Camera(camera_configs[0]), Camera(camera_configs[1]), Camera(camera_configs[2]), Camera(camera_configs[3])})
{
    
    _UpdateCamStatus();
    SPDLOG_INFO("Camera Manager initialized");
}

void CameraManager::_UpdateCamStatus()
{
    // Initialize cam_status based on the status of each camera
    for (size_t i = 0; i < NUM_CAMERAS; ++i) {
        cam_status[i] = cameras[i].GetCamStatus();
    }

}

bool CameraManager::_UpdateCameraConfigs(Payload* payload)
{
    bool config_changed = false;

    // Check if the configuration of the cameras has changed
    for (auto& camera : cameras) 
    {

        // Check status of each camera and modify configuration accordingly
        auto config = GetCameraConfig(camera.GetCamId());
        if (camera.IsEnabled() != config->enable) {
            config_changed = true;
            config->enable = camera.IsEnabled();
        }
    }

    if (config_changed) 
    {
        SPDLOG_WARN("Camera configuration modified");
        assert(payload != nullptr); // TODO
        payload->GetConfiguration().UpdateCameraConfigs(camera_configs);
    }

    return config_changed;
}


void CameraManager::TurnOn()
{
    for (auto& camera : cameras) 
    {
        if (camera.IsEnabled())
        { 
            camera.TurnOn();
        }
    }

    _UpdateCamStatus();
}

void CameraManager::TurnOff()
{
    for (auto& camera : cameras) 
    {
        if (camera.IsEnabled())
        { 
            camera.TurnOff();
        }
    }

    _UpdateCamStatus();
}


void CameraManager::CaptureFrames(std::vector<bool>& captured_flags)
{

    for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        if (cameras[i].GetCamStatus() == CAM_STATUS::TURNED_ON)
        {
            captured_flags[i] = cameras[i].CaptureFrame();
        }
    }

}

void CameraManager::CaptureFrames()
{
    std::vector<bool> captured_flags(NUM_CAMERAS, false);
    CaptureFrames(captured_flags);
}


uint8_t CameraManager::SaveLatestFrames(std::vector<bool>& captured_flags)
{
    uint8_t save_count = 0;
    for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        if (captured_flags[i])
        {
            // TODO Save to disk
            save_count++;
        }
    }
    return save_count;
}

uint8_t CameraManager::SaveLatestFrames()
{
    std::vector<bool> captured_flags(NUM_CAMERAS, false);
    return SaveLatestFrames(captured_flags);
}



void CameraManager::RunLoop(Payload* payload)
{
    loop_flag = true;

    auto current_capture_time = std::chrono::high_resolution_clock::now();
    auto last_capture_time = std::chrono::high_resolution_clock::now();

    std::vector<bool> captured_flags(NUM_CAMERAS, false);

    while (loop_flag) 
    {

        // TODO remove busy waiting

        // reset captured flag to all false
        std::fill(captured_flags.begin(), captured_flags.end(), false);

        // Capture frames for each turned on camera - TODO thread protection
        CaptureFrames(captured_flags);


        switch (capture_mode)
        {
            
            case CAPTURE_MODE::IDLE:
            {
                break;
            }
            

            case CAPTURE_MODE::CAPTURE_SINGLE: // Response to a command
            {
                
                SaveLatestFrames(captured_flags);
                SPDLOG_INFO("Single capture request completed");
                // TODO should be a way to ACK the command here 
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
                    periodic_frames_captured += SaveLatestFrames(captured_flags);
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
                break;
            }

            case CAPTURE_MODE::VIDEO_STREAM:
            {
                break;
            }

            default:
                SPDLOG_WARN("Unknown capture mode: {}", capture_mode.load());
                break;
        }

        _UpdateCamStatus();
        _UpdateCameraConfigs(payload);

    }

    SPDLOG_INFO("Exiting Camera Manager Run Loop");
    SetDisplayFlag(false);

}

void CameraManager::RunDisplayLoop()
{

    int active_cams = 0;
    
    while (display_flag && loop_flag) 
    {
        

        active_cams = 0;
        for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
        {
            if (cameras[i].GetCamStatus() == CAM_STATUS::TURNED_ON)
            {
                cameras[i].DisplayLastFrame();
                active_cams++;
            }
        }

        if (active_cams == 0) 
        {
            SPDLOG_WARN("No cameras are turned on. Exiting display loop.");
            break;
        }

        if (!display_flag || !loop_flag) 
        {
            SPDLOG_WARN("Display loop terminated by flags.");
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
        // SPDLOG_WARN("Address of display flag: {}", static_cast<const void*>(&display_flag));
    }

    display_flag = false;
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
    display_flag = false;
    loop_flag = false;
    std::cout << "DISPLAY LOOP STOPPED " << display_flag << std::endl; 
}

void CameraManager::SetDisplayFlag(bool display_flag)
{
    this->display_flag = display_flag;
}

bool CameraManager::GetDisplayFlag() const
{
    return display_flag;
}

void CameraManager::SetCaptureMode(CAPTURE_MODE mode)
{
    capture_mode = mode;
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
        if (camera.GetCamId() == cam_id)
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
        if (camera.GetCamId() == cam_id)
        {
            return camera.Disable();
        }
    }

    return false;
}