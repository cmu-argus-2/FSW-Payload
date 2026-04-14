#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <atomic>
#include <optional>
#include <string>
#include <thread>
#include <shared_mutex>
#include <utility>
#include "frame.hpp"
#include <opencv2/opencv.hpp>
#include "core/errors.hpp"

#define DEFAULT_CAMERA_FPS 10
#define MAX_CONSECUTIVE_ERROR_COUNT 3

// ISP parameters shared across all cameras (nvarguscamerasrc properties).
// Defined here so Camera can store a copy; camera_manager.hpp re-exports it via camera.hpp.
struct CameraISPConfig
{
    int   wbmode               = 0;     // white balance mode (0 = auto)
    bool  aelock               = false; // auto-exposure lock
    bool  awblock              = false; // auto-white-balance lock
    int   ee_mode              = 1;     // edge enhancement mode (1 = EdgeEnhance_Fast)
    float ee_strength          = -1.0f; // edge enhancement strength (-1 = driver default)
    int   aeantibanding        = 1;     // AE antibanding mode (1 = auto)
    float exposurecompensation = 0.0f;  // EV compensation in stops
    int   tnr_mode             = 1;     // temporal noise reduction mode (1 = TNR_Fast)
    float tnr_strength         = -1.0f; // TNR strength (-1 = driver default)
    float saturation           = 1.0f;  // colour saturation multiplier
    int   fps                  = DEFAULT_CAMERA_FPS;
    int   max_buffers          = 2;     // appsink max-buffers
    // Optional range properties — absent means the property is not set in the pipeline
    std::optional<std::pair<int64_t, int64_t>> exposuretimerange;  // nanoseconds [min, max]
    std::optional<std::pair<float, float>>     gainrange;           // sensor gain [min, max]
    std::optional<std::pair<float, float>>     ispdigitalgainrange; // ISP digital gain [min, max]
};


enum class CAM_STATUS : uint8_t 
{
    UNDEFINED = 0,
    INACTIVE = 1,
    ACTIVE = 2,
};

enum class CAM_ERROR : uint8_t
 {
    NO_ERROR = 0x00,
    CAPTURE_FAILED = 0x01,
    INITIALIZATION_FAILED = 0x02
};

class Camera
{


public:
    Camera(int id, std::string path, const CameraISPConfig& isp);

    // Attempt to enable the camera. Returns true if successful
    bool Enable();
    // Disable the camera. Returns true if successful
    bool Disable();
    bool IsEnabled() const;

    // void Restart(); // Restart after failure, but need to avoid loops

    void CaptureFrame();
    const Frame& GetBufferFrame() const;
    void CopyBufferFrame(Frame& dest) const;
    void SetOffNewFrameFlag();
    bool IsNewFrameAvailable() const;

    void RunCaptureLoop();
    void StopCaptureLoop();
    bool IsCaptureLoopRunning() const;
    
    int GetID() const;
    CAM_STATUS GetStatus() const;
    CAM_ERROR GetLastError() const;


    void DisplayLastFrame();

    void LoadIntrinsics(const cv::Mat& intrinsics, const cv::Mat& distortion_parameters);


private:

    int width;
    int height;
    CameraISPConfig isp_config;

    std::atomic<CAM_STATUS> cam_status;
    CAM_ERROR last_error;
    int consecutive_error_count;
    void HandleErrors(CAM_ERROR error);
    EC MapError(CAM_ERROR cam_error);
    
    
    int cam_id;
    std::string cam_path;
    cv::VideoCapture cap;
    cv::Mat local_buffer_img;


    Frame buffer_frame;
    std::atomic<bool> _new_frame_flag;

    std::thread capture_thread;
    mutable std::shared_mutex frame_mutex;
    std::atomic<bool> capture_loop_flag = false;

    cv::Mat intrinsics;
    cv::Mat distortion_parameters; 


};










#endif // CAMERA_HPP