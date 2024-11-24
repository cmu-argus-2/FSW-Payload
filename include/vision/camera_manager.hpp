#ifndef CAMERA_MANAGER_HPP
#define CAMERA_MANAGER_HPP

#include <array>
#include <atomic>
#include "spdlog/spdlog.h"
#include "camera.hpp"
#include "configuration.hpp"

#define NUM_CAMERAS 4
#define MAX_PERIODIC_FRAMES_TO_CAPTURE 255
#define DEFAULT_PERIODIC_FRAMES_TO_CAPTURE 100

// Forward declaration of Payload class
class Payload;

// Capture modes for the camera system
enum class CAPTURE_MODE : uint8_t {
    IDLE = 0,            // Camera system is not saving frames and waiting for a command
    CAPTURE_SINGLE = 1,  // Camera system stores the latest frame from each available cameras
    PERIODIC = 2,        // Camera system stores all frames at a fixed rate
    PERIODIC_EARTH = 3,  // Camera system store frames at a fixed rate, but applies a filter to store only frames with a visible Earth
    VIDEO_STREAM = 4     // Camera system store frames at a fixed rate and streams them to a video file
};


// Main interface to manage the cameras 
class CameraManager
{

public:

    CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs);


    void TurnOn();
    void TurnOff();

    void RunLoop(Payload* payload);
    void StopLoops();
    void SetDisplayFlag(bool display_flag);
    bool GetDisplayFlag() const;
    void RunDisplayLoop(); // This will block the calling thread until the display flag is set to false or all cameras are turned off

    CameraConfig* GetCameraConfig(int cam_id);

    void CaptureFrames();

    uint8_t SaveLatestFrames();


    // Set the capture mode of the camera system
    void SetCaptureMode(CAPTURE_MODE mode);
    // Send a capture request to the cameras. Wrapper around SetCaptureMode for CAPTURE_SINGLE 
    void SendCaptureRequest();
    // Set the rate at which the camera system captures frames in PERIODIC mode
    void SetPeriodicCaptureRate(uint8_t rate);
    // Set the number of frames to capture in PERIODIC mode
    void SetPeriodicFramesToCapture(uint8_t frames);


    void EnableCameras(std::vector<int>& id_activated_cams);
    void DisableCameras(std::vector<int>& id_disabled_cams);


    bool EnableCamera(int cam_id);
    bool DisableCamera(int cam_id);

private:
        
    std::atomic<CAPTURE_MODE> capture_mode;

    std::array<CameraConfig, NUM_CAMERAS> camera_configs;
    std::array<Camera, NUM_CAMERAS> cameras;
    
    bool config_changed = false;

    std::atomic<bool> display_flag = false;
    std::atomic<bool> loop_flag = false;
    
    std::atomic<uint8_t> periodic_capture_rate = 5; // Default rate of 5 seconds
    std::atomic<uint8_t> periodic_frames_to_capture = DEFAULT_PERIODIC_FRAMES_TO_CAPTURE; // After the request is serviced, it gets back to the default value
    std::atomic<uint8_t> periodic_frames_captured = 0;


    std::array<CAM_STATUS, NUM_CAMERAS> cam_status;

    void _UpdateCamStatus();

    // returns True if the camera configurations have been updated, False otherwise
    bool _UpdateCameraConfigs(Payload* payload);




};







#endif // CAMERA_MANAGER_HPP