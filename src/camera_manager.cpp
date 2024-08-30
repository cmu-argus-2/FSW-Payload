#include "camera_manager.hpp"



CameraManager::CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs) 
:
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


void CameraManager::RunLoop()
{
    loop_flag = true;
    std::vector<int> active_camera_ids; 

    // TODO need to check status of each camera and modify config accordingly 

    while (loop_flag) 
    {
        for (auto& camera : cameras) 
        {
            bool captured = false;
            
            if (camera.IsEnabled())
            {
                captured = camera.CaptureFrame();
            }

            if (display_flag)
            {
                if (camera.IsEnabled() && captured)
                {
                    cv::imshow("Camera " + std::to_string(camera.GetBufferFrame().GetCamId()), camera.GetBufferFrame().GetImg());
                }
            }

            
            if (camera.IsEnabled())
            {
                active_camera_ids.push_back(camera.GetCamId());
            }

        }

        cv::waitKey(1);
        SPDLOG_INFO("IDs of active cameras: {}", fmt::format("{}", fmt::join(active_camera_ids, ", ")));
        active_camera_ids.clear();
    }

}

void CameraManager::StopLoop()
{
    this->loop_flag = false;
}

void CameraManager::DisplayLoop(bool display_flag)
{
    this->display_flag = display_flag;
}

