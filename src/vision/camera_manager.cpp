#include "vision/camera_manager.hpp"
#include "core/data_handling.hpp"
#include "payload.hpp"

CameraManager::CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs) 
:
capture_mode(CAPTURE_MODE::IDLE),
camera_configs(camera_configs),
cameras({Camera(camera_configs[0].id, camera_configs[0].path), 
    Camera(camera_configs[1].id, camera_configs[1].path), 
    Camera(camera_configs[2].id, camera_configs[2].path), 
    Camera(camera_configs[3].id, camera_configs[3].path)})
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


uint8_t CameraManager::SaveLatestFrames()
{
    uint8_t save_count = 0;
    for (std::size_t i = 0; i < NUM_CAMERAS; ++i) 
    {
        if (cameras[i].GetStatus() == CAM_STATUS::ACTIVE && cameras[i].IsNewFrameAvailable())
        {
            DH::StoreRawImgToDisk(
                cameras[i].GetBufferFrame()._timestamp,
                cameras[i].GetBufferFrame()._cam_id,
                cameras[i].GetBufferFrame()._img
            );
            cameras[i].SetOffNewFrameFlag();
            save_count++;
        }  
    }
    return save_count;
}



void CameraManager::RunLoop(Payload* payload)
{
    loop_flag = true;

    auto last_health_check_time = std::chrono::high_resolution_clock::now(); // Track health check timing
    auto current_capture_time = std::chrono::high_resolution_clock::now();
    auto last_capture_time = std::chrono::high_resolution_clock::now();

    while (loop_flag) 
    {
        // TODO remove busy waiting ~ add condition variable
        switch (capture_mode.load())
        {
            case CAPTURE_MODE::IDLE:
            {
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
                    periodic_frames_captured += SaveLatestFrames();
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

            default:
                SPDLOG_WARN("Unknown capture mode: {}", capture_mode.load());
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
    
    while (display_flag && loop_flag) 
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

        if (!display_flag || !loop_flag) 
        {
            SPDLOG_WARN("Display loop terminated by flags.");
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
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

    for (auto& camera : cameras) 
    {
        camera.StopCaptureLoop();
    }

    SPDLOG_INFO("Stopped camera loops...");
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


void CameraManager::EnableCameras(std::vector<int>& id_activated_cams)
{
    for (auto& camera : cameras) 
    {
        camera.Enable();

        if (camera.GetStatus() == CAM_STATUS::ACTIVE)
        {
            id_activated_cams.push_back(camera.GetID());
        }
    }

    _UpdateCamStatus();
}


void CameraManager::DisableCameras(std::vector<int>& id_disabled_cams)
{
    for (auto& camera : cameras) 
    {
        camera.Disable();

        if (camera.GetStatus() == CAM_STATUS::INACTIVE)
        {
            id_disabled_cams.push_back(camera.GetID());
        }
    }

    _UpdateCamStatus();
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