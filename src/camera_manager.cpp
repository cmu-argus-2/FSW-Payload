#include "camera_manager.hpp"
#include "payload.hpp"



CameraManager::CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs) 
:
camera_configs(camera_configs),
cameras({Camera(camera_configs[0]), Camera(camera_configs[1]), Camera(camera_configs[2]), Camera(camera_configs[3])})
{
    SPDLOG_INFO("Camera Manager initialized");
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
}


void CameraManager::RunLoop(Payload* payload)
{
    loop_flag = true;
    std::vector<int> active_camera_ids; 


    while (loop_flag) 
    {
        
        for (auto& camera : cameras) 
        {
            bool captured = false;
            
            if (camera.GetCamStatus() == CAM_STATUS::TURNED_ON)
            {
                captured = camera.CaptureFrame();
            }

            if (display_flag && captured)
            {
                cv::imshow("Camera " + std::to_string(camera.GetBufferFrame().GetCamId()), camera.GetBufferFrame().GetImg());
            }

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

        if (config_changed) {
            SPDLOG_WARN("Camera configuration modified");
            assert(payload != nullptr); // TODO
            payload->GetConfiguration().UpdateCameraConfigs(camera_configs);
            config_changed = false;
        }

        cv::waitKey(1);

        SPDLOG_INFO("IDs of active cameras: {}", fmt::format("{}", fmt::join(active_camera_ids, ", ")));
        active_camera_ids.clear();
    }

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

