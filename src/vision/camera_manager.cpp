#include "vision/camera_manager.hpp"
#include "payload.hpp"

CameraManager::CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs) 
:
capture_mode(CAPTURE_MODE::IDLE),
camera_configs(camera_configs),
cameras({Camera(camera_configs[0]), Camera(camera_configs[1]), Camera(camera_configs[2]), Camera(camera_configs[3])})
{
    
    UpdateCamStatus();
    SPDLOG_INFO("Camera Manager initialized");
}

void CameraManager::UpdateCamStatus()
{
    // Initialize cam_status based on the status of each camera
    for (size_t i = 0; i < NUM_CAMERAS; ++i) {
        cam_status[i] = cameras[i].GetCamStatus();
    }

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

    UpdateCamStatus();
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

    UpdateCamStatus();
}


void CameraManager::RunLoop(Payload* payload)
{
    loop_flag = true;
    std::vector<int> active_camera_ids; 


    while (loop_flag) 
    {
        // TODO: temporary
        // SetCaptureMode(CAPTURE_MODE::CAPTURE_SINGLE);
        switch (capture_mode)
        {
            case CAPTURE_MODE::IDLE:
                break;


            case CAPTURE_MODE::CAPTURE_SINGLE: // Response to a command

                for (auto& camera : cameras) 
                {
                    bool captured = false;
                    if (camera.GetCamStatus() == CAM_STATUS::TURNED_ON)
                    {
                        captured = camera.CaptureFrame();

                        // For the debugging - display
                        if (display_flag && captured)
                        {
                            camera.DisplayLastFrame();
                        }
                    }
                }


                // TODO should be a way to ACK the command here 
                SetCaptureMode(CAPTURE_MODE::IDLE);
                break;


            case CAPTURE_MODE::PERIODIC:
                break;


            case CAPTURE_MODE::PERIODIC_EARTH:
                break;


            case CAPTURE_MODE::VIDEO_STREAM:
                break;


        }


        // Check if the configuration of the cameras has changed
        for (auto& camera : cameras) 
        {
            // Separated since the status might change after the capture attempt
            if (camera.GetCamStatus() == CAM_STATUS::TURNED_ON)
            {
                active_camera_ids.push_back(camera.GetCamId());
            }

            // Check status of each camera and modify configuration accordingly
            auto config = GetCameraConfig(camera.GetCamId());
            if (camera.IsEnabled() != config->enable) {
                config_changed = true;
                config->enable = camera.IsEnabled();
            }
        }


        UpdateCamStatus();

        if (config_changed) {
            SPDLOG_WARN("Camera configuration modified");
            assert(payload != nullptr); // TODO
            payload->GetConfiguration().UpdateCameraConfigs(camera_configs);
            config_changed = false;
        }

        if (display_flag)
        {
            cv::waitKey(1);
        }
        // SPDLOG_INFO("IDs of active cameras: {}", fmt::format("{}", fmt::join(active_camera_ids, ", ")));
        active_camera_ids.clear();

    }

    SPDLOG_INFO("Exiting Camera Manager Run Loop");

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

void CameraManager::StopLoop()
{
    this->loop_flag = false;
}

void CameraManager::DisplayLoop(bool display_flag)
{
    this->display_flag = display_flag;
}
void CameraManager::SetCaptureMode(CAPTURE_MODE mode)
{
    capture_mode = mode;
}

void CameraManager::SendCaptureRequest()
{
    SetCaptureMode(CAPTURE_MODE::CAPTURE_SINGLE);
}