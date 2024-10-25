#ifndef CAMERA_MANAGER_HPP
#define CAMERA_MANAGER_HPP

#include <array>
#include "spdlog/spdlog.h"
#include "camera.hpp"
#include "configuration.hpp"

#define NUM_CAMERAS 4

// Forward declaration of Payload class
class Payload;

// Main interface to manage the cameras 
class CameraManager
{

public:

    CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs);


    void TurnOn();
    void TurnOff();

    void RunLoop(Payload* payload);
    void StopLoop();
    void DisplayLoop(bool display_flag);

    CameraConfig* GetCameraConfig(int cam_id);

    void GetStatus();


private:
        
    std::array<CameraConfig, NUM_CAMERAS> camera_configs;
    std::array<Camera, NUM_CAMERAS> cameras;
    
    bool config_changed = false;

    std::atomic<bool> display_flag = false;
    std::atomic<bool> loop_flag = false;


    std::array<CAM_STATUS, NUM_CAMERAS> cam_status;
    void UpdateCamStatus();
};







#endif // CAMERA_MANAGER_HPP