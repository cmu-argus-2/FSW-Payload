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

    while (loop_flag) 
    {
        for (auto& camera : cameras) 
        {
            if (camera.IsEnabled())
            {
                camera.CaptureFrame();
            }
        }

        if (display_flag) 
        {
            for (auto& camera : cameras) 
            { 
                if (camera.IsEnabled() && !camera.GetBufferFrame().GetImg().empty()) 
                {
                    cv::imshow("Camera " + std::to_string(camera.GetBufferFrame().GetCamId()), camera.GetBufferFrame().GetImg());
                }
            }
        }

        cv::waitKey(1);
            
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

