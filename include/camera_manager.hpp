#ifndef CAMERA_MANAGER_HPP
#define CAMERA_MANAGER_HPP

#include <array>
#include "spdlog/spdlog.h"
#include "camera.hpp"
#include "configuration.hpp"

#define NUM_CAMERAS 4

// Main interface to manage the cameras 

class CameraManager
{


public:

    CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs);


    void TurnOn();
    void TurnOff();

    void RunLoop();
    void StopLoop();
    void DisplayLoop(bool display_flag);


    CameraConfig* GetCameraConfig(int cam_id);



    

private:
        
    std::array<Camera, NUM_CAMERAS> cameras;
    std::array<CameraConfig, NUM_CAMERAS> camera_configs;

    
    bool config_changed = false;

    std::atomic<bool> display_flag = false;
    std::atomic<bool> loop_flag = false;

};







#endif // CAMERA_MANAGER_HPP