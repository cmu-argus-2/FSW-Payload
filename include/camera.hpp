#ifndef CAMERA_HPP
#define CAMERA_HPP


#include <mutex>
#include <atomic>
#include "frame.hpp"
#include <opencv2/opencv.hpp>
#include "configuration.hpp"


#define DEFAULT_CAMERA_WIDTH 640
#define DEFAULT_CAMERA_HEIGHT 480

#define MAX_CONSECUTIVE_ERROR_COUNT 3 // before disabling 



enum class CAM_STATUS : uint8_t {
    DISABLED = 0x00,
    TURNED_ON = 0x01,
    TURNED_OFF = 0x02
};

enum class CAM_ERROR : uint8_t {
    NO_ERROR = 0x00,
    CAPTURE_FAILED = 0x01,
    INITIALIZATION_FAILED = 0x02
};

class Camera
{


public:
    Camera(int id, std::string path, bool enabled = false);
    Camera(const CameraConfig& config);


    void TurnOn();
    void TurnOff();
    // void Restart(); // Restart after failure, but need to avoid loops

    bool CaptureFrame();
    const Frame& GetBufferFrame() const;

    bool IsEnabled() const;
    const int GetCamId() const;
    CAM_STATUS GetCamStatus() const;


    void LoadIntrinsics(const cv::Mat& intrinsics, const cv::Mat& distortion_parameters);


private:

    CAM_STATUS cam_status;
    CAM_ERROR last_error;
    
    std::string cam_path;
    int cam_id;
    cv::VideoCapture cap;


    Frame buffer_frame;

    cv::Mat intrinsics;
    cv::Mat distortion_parameters; 



    int consecutive_error_count = 0;
    void HandleErrors(CAM_ERROR error);


};










#endif // CAMERA_HPP