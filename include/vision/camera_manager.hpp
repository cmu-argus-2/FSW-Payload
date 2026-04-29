#ifndef CAMERA_MANAGER_HPP
#define CAMERA_MANAGER_HPP

#include <array>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "spdlog/spdlog.h"
#include "camera.hpp"  // also defines CameraISPConfig

#define NUM_CAMERAS 4
#define MAX_PERIODIC_FRAMES_TO_CAPTURE 255
#define DEFAULT_PERIODIC_FRAMES_TO_CAPTURE 100
#define CAMERA_HEALTH_CHECK_INTERVAL 500 // seconds

class InferenceManager; // forward declaration

// Per-camera identity and resolution
struct CameraConfig
{
    int64_t id;
    std::string path;
    int64_t width;
    int64_t height;
    bool enabled = true;
};



// Capture modes for the camera system
enum class CAPTURE_MODE : uint8_t {
    IDLE = 0,            // Camera system is not saving frames and waiting for a command
    CAPTURE_SINGLE = 1,  // Camera system stores the latest frame from each available cameras
    PERIODIC = 2,        // Camera system stores all frames at a fixed frequency
    PERIODIC_EARTH = 3,  // Camera system store frames at a fixed rate, but applies a filter to store only frames with a visible Earth
    PERIODIC_ROI = 4, // Camera system store frames at a fixed rate, but only stores frames with regions of interest
    PERIODIC_LDMK = 5, // Camera system store frames at a fixed rate, but only stores frames with detected landmarks
};

// Returns the ProcessingStage implied by a given capture mode,
// i.e. what has already been done to frames stored by that mode.
inline ProcessingStage CaptureModeToProcessingStage(CAPTURE_MODE mode)
{
    switch (mode)
    {
        case CAPTURE_MODE::PERIODIC_LDMK:  return ProcessingStage::LDNeted;
        case CAPTURE_MODE::PERIODIC_ROI:   return ProcessingStage::RCNeted;
        case CAPTURE_MODE::PERIODIC_EARTH: return ProcessingStage::Prefiltered;
        default:                           return ProcessingStage::NotPrefiltered;
    }
}


// Main interface to manage the cameras 
class CameraManager
{

public:

    CameraManager(const std::array<CameraConfig, NUM_CAMERAS>& camera_configs,
                  const CameraISPConfig& isp_config,
                  InferenceManager& inference_manager);
    ~CameraManager();

    void RunLoop();
    void StopLoops();
    void SetDisplayFlag(bool display_flag);
    bool GetDisplayFlag() const;
    void RunDisplayLoop(); // This will block the calling thread until the display flag is set to false or all cameras are turned off

    CameraConfig* GetCameraConfig(int cam_id);
    int GetCapturedFramesCount() const;
    std::vector<std::tuple<uint8_t, uint64_t>> GetBufferFrameIDs() const; // non-destructive read
    std::vector<std::tuple<uint8_t, uint64_t>> DrainBufferFrameIDs();     // returns and clears
    void ResetCaptureState();  // zeros periodic_frames_captured and clears buffer_frame_ids atomically

    void CaptureFrames();

    uint8_t SaveLatestFrames(CAPTURE_MODE mode = CAPTURE_MODE::PERIODIC);

    // Copy the latest frames from each camera to the provided vector and return the number of frames copied
    uint8_t CopyFrames(std::vector<Frame>& vec_frames, bool only_earth = true);


    // Set the capture mode of the camera system
    void SetCaptureMode(CAPTURE_MODE mode);
    // Send a capture request to the cameras. Wrapper around SetCaptureMode for CAPTURE_SINGLE 
    bool SendCaptureRequest();
    // Set the rate at which the camera system captures frames in PERIODIC mode
    void SetPeriodicCaptureRate(uint8_t rate);
    // Set the number of frames to capture in PERIODIC mode
    void SetPeriodicFramesToCapture(uint8_t frames);
    // Set the folder to store the captured images
    void SetStorageFolder(const std::string& folder);
    void SetTargetProcessingStage(ProcessingStage stage);

    std::string GetStorageFolder();
    ProcessingStage GetTargetProcessingStage() const;

    // Returns the number of cameras that were activated/disabled
    int EnableCameras();
    int DisableCameras();

    bool EnableCamera(int cam_id);
    bool DisableCamera(int cam_id);

    // Set which cameras EnableCameras() may activate. Cameras absent from the mask
    // are disabled immediately and skipped by EnableCameras() until the mask is cleared.
    // Call at dataset collection start; call ClearDatasetCamerasMask at stop.
    void SetDatasetCamerasMask(const std::array<bool, NUM_CAMERAS>& mask);
    void ClearDatasetCamerasMask();

    CAPTURE_MODE GetCaptureMode() const;
    int CountActiveCameras() const;
    int CountConfiguredCameras() const; // cameras with enabled=true in config
    void FillCameraStatus(uint8_t* status);

    bool PrepareForCapture();

private:
        
    InferenceManager& inferenceManager;
    CameraISPConfig isp_config;

    std::atomic<CAPTURE_MODE> capture_mode;
    std::string storage_folder;
    std::mutex storage_folder_m;

    std::array<CameraConfig, NUM_CAMERAS> camera_configs;
    std::array<Camera, NUM_CAMERAS> cameras;

    std::atomic<bool> display_flag = false;
    std::atomic<bool> loop_flag = false;
    std::atomic<bool> auto_disable_after_capture = false;

    std::mutex capture_mode_mutex;
    std::condition_variable capture_mode_cv;
    
    std::atomic<uint8_t> periodic_capture_rate = 60; // Default rate of 60 seconds
    std::atomic<uint8_t> periodic_frames_to_capture = DEFAULT_PERIODIC_FRAMES_TO_CAPTURE; // After the request is serviced, it gets back to the default value
    std::atomic<uint8_t> periodic_frames_captured = 0;
    std::atomic<ProcessingStage> target_processing_stage = ProcessingStage::NotPrefiltered;

    // Buffer to store the latest frame IDs (cam_id, timestamp) for each camera.
    std::vector<std::tuple<uint8_t, uint64_t>> buffer_frame_ids;
    mutable std::mutex buffer_frame_ids_m;


    std::array<CAM_STATUS, NUM_CAMERAS> cam_status;

    std::array<bool, NUM_CAMERAS> dataset_camera_mask = {true, true, true, true};
    std::mutex dataset_camera_mask_m;


    void _PerformCameraHealthCheck(); // background watchdog for the cameras
    void _UpdateCamStatus();
    void _AutoDisableIfNeeded();



};







#endif // CAMERA_MANAGER_HPP
