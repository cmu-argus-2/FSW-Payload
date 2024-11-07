#ifndef CAMERA_MANAGER_HPP
#define CAMERA_MANAGER_HPP

#include <array>
#include "spdlog/spdlog.h"
#include "camera.hpp"
#include "configuration.hpp"

#define NUM_CAMERAS 4

// Forward declaration of Payload class
class Payload;

// Capture modes for the camera system
enum class CAPTURE_MODE : uint8_t {
    IDLE = 0,            // Camera system is not capturing frames and waiting for a command
    CAPTURE_SINGLE = 1,  // Camera system captures and stores the latest frame from each available cameras
    PERIODIC = 2,        // Camera system captures and stores all frames at a fixed rate
    PERIODIC_EARTH = 3,  // Camera system captures frames at a fixed rate, but applies a filter to store only frames with a visible Earth
    VIDEO_STREAM = 4     // Camera system captures frames at a fixed rate and streams them to a video file
};


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



    // Set the capture mode of the camera system
    void SetCaptureMode(CAPTURE_MODE mode);
    // Send a capture request to the cameras. Wrapper around SetCaptureMode for CAPTURE_SINGLE 
    void SendCaptureRequest();
    // Set the rate at which the camera system captures frames in PERIODIC mode
    void SetPeriodicCaptureRate(int rate);


    void GetStatus();


private:
        
    std::atomic<CAPTURE_MODE> capture_mode;

    std::array<CameraConfig, NUM_CAMERAS> camera_configs;
    std::array<Camera, NUM_CAMERAS> cameras;
    
    bool config_changed = false;

    std::atomic<bool> display_flag = false;
    std::atomic<bool> loop_flag = false;
    
    std::atomic<int> periodic_capture_rate = 5; // Default rate of 5 seconds


    std::array<CAM_STATUS, NUM_CAMERAS> cam_status;
    void UpdateCamStatus();

};







#endif // CAMERA_MANAGER_HPP