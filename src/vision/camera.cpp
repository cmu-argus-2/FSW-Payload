#include <cstdint>
#include <chrono>
#include "spdlog/spdlog.h"
#include "vision/camera.hpp"


Camera::Camera(int cam_id, std::string path, bool enabled)
: 
cam_status((enabled == false) ? CAM_STATUS::DISABLED : CAM_STATUS::TURNED_OFF),
last_error(CAM_ERROR::NO_ERROR),
consecutive_error_count(0),
cam_id(cam_id),
cam_path(path),
buffer_frame(cam_id, cv::Mat(), 0),
_capture_loop(false),
_new_frame_flag(false)
{
}

Camera::Camera(const CameraConfig& config)
:
cam_status((config.enable == false) ? CAM_STATUS::DISABLED : CAM_STATUS::TURNED_OFF),
last_error(CAM_ERROR::NO_ERROR),
consecutive_error_count(0),
cam_id(static_cast<int>(config.id)),
cam_path(config.path),
buffer_frame(cam_id, cv::Mat(height, width, CV_8UC3), 0),
_capture_loop(false),
_new_frame_flag(false)
{
}

bool Camera::Enable()
{

    if (cam_status == CAM_STATUS::TURNED_ON) 
    {
        SPDLOG_WARN("CAM{}: Camera is already enabled", cam_id);
        return true;
    } 
    else if (cam_status == CAM_STATUS::DISABLED) 
    {
        cam_status = CAM_STATUS::TURNED_OFF;
       
    }

    // attempt to turn on the camera
    TurnOn();

    if (cam_status == CAM_STATUS::TURNED_ON) 
    {
        SPDLOG_INFO("CAM{}: Camera enabled", cam_id);
        return true;
    } 
    else 
    {
        SPDLOG_ERROR("CAM{}: Camera failed to enable", cam_id);
        return false;
    }
}

bool Camera::Disable()
{
    if (cam_status == CAM_STATUS::DISABLED) 
    {
        SPDLOG_WARN("CAM{}: Camera is already disabled", cam_id);
        return true;
    } 
    else if (cam_status == CAM_STATUS::TURNED_OFF) 
    {
        cam_status = CAM_STATUS::DISABLED;
        SPDLOG_INFO("CAM{}: Camera disabled", cam_id);
        return true;
    }

    // attempt to turn off the camera
    TurnOff();

    if (cam_status == CAM_STATUS::DISABLED) 
    {
        SPDLOG_INFO("CAM{}: Camera disabled", cam_id);
        return true;
    } 
    else 
    {
        SPDLOG_ERROR("CAM{}: Camera failed to disable", cam_id);
        return false;
    }
}




void Camera::TurnOn()
{
    try {

        if (cam_status == CAM_STATUS::DISABLED) {
            SPDLOG_ERROR("CAM{}: Camera is disabled", cam_id);
            return;
        }

        cap.open(cam_path, cv::CAP_V4L2);

        // Check if the camera is opened successfully
        if (!cap.isOpened()) {
            SPDLOG_ERROR("Unable to open the camera");
            throw std::runtime_error("Unable to open the camera");
        }

        // Set the camera resolution - remove by reading confif file
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap.set(cv::CAP_PROP_FPS, DEFAULT_CAMERA_FPS);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 3);
        cap.set(cv::CAP_PROP_GAIN, 10000); // Set gain to maximum - OpenCV will clamp it to the maximum value
        SPDLOG_INFO("CAM{}: Camera gain set to {}", cam_id, cap.get(cv::CAP_PROP_GAIN));

        cam_status = CAM_STATUS::TURNED_ON;
        SPDLOG_INFO("CAM{}: Camera successfully turned on", cam_id);


    } catch (const std::exception& e) {
        SPDLOG_ERROR("Exception occurred: {}", e.what());


        HandleErrors(CAM_ERROR::INITIALIZATION_FAILED);

        // only be necessary if the camera was opened but failed afterwards
        if (cap.isOpened()) {
            cap.release();
        }

    }

}

void Camera::TurnOff()
{
    if (cam_status == CAM_STATUS::DISABLED) {
        SPDLOG_ERROR("CAM{}: Camera is disabled", cam_id);
        return;
    }

    if (cam_status == CAM_STATUS::TURNED_OFF) {
        SPDLOG_ERROR("CAM{}: Camera is already turned off", cam_id);
        return;
    }
    

    if (cam_status == CAM_STATUS::TURNED_ON) {
        cap.release();
        cam_status = CAM_STATUS::TURNED_OFF;
    }
    

}

bool Camera::CaptureFrame()
{
    // Single responsibility principle. Status must be checked externally

    static cv::Mat captured_frame(height, width, CV_8UC3); // Preallocated buffer

    cap >> captured_frame;

    if (captured_frame.empty()) {
        SPDLOG_ERROR("Unable to capture frame");
        HandleErrors(CAM_ERROR::CAPTURE_FAILED);
        return false;
    }
    
    // Capture current timestamp - TODO access the reference from the hardware API (OpenCV is garbage)
    buffer_frame._timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();


   // Update buffer_frame attributes in place
    buffer_frame._img = captured_frame.clone(); // Deep copy, but pretty bad in terms of memory usage given our resolutions.. TODO: Optimize
    buffer_frame._cam_id = cam_id;

    _new_frame_flag = true;


    return _new_frame_flag;
}


void Camera::SetOffNewFrameFlag()
{
    _new_frame_flag = false;
}

bool Camera::IsNewFrameAvailable() const
{
    return _new_frame_flag;
}

void Camera::HandleErrors(CAM_ERROR error)
{
    last_error = error;

    if (last_error == CAM_ERROR::NO_ERROR) {
        consecutive_error_count = 0;
        return;
    }

    consecutive_error_count++;
    SPDLOG_ERROR("CAM{}: Error occurred: {}", cam_id, last_error);

    if (consecutive_error_count >= MAX_CONSECUTIVE_ERROR_COUNT) {
        cam_status = CAM_STATUS::DISABLED;
        SPDLOG_ERROR("CAM{}: Camera disabled after {} errors", cam_id, consecutive_error_count);
        // consecutive_error_count = 0; // Reset the count for next retry? // Could be stuck in a loop
        return;
    }

    if (last_error == CAM_ERROR::INITIALIZATION_FAILED) {
        cam_status = CAM_STATUS::DISABLED;
        SPDLOG_ERROR("CAM{}: Initialization failed. Still disabled.", cam_id);
        return;
    }

}


void Camera::LoadIntrinsics(const cv::Mat& intrinsics, const cv::Mat& distortion_parameters)
{
    this->intrinsics = intrinsics.clone();
    this->distortion_parameters = distortion_parameters.clone();
}


const Frame& Camera::GetBufferFrame() const
{
    return buffer_frame;
}

bool Camera::IsEnabled() const
{
    return (cam_status == CAM_STATUS::DISABLED) ? false : true;
}

int Camera::GetCamId() const
{
    return cam_id;
}

CAM_STATUS Camera::GetCamStatus() const
{
    return cam_status;
}

void Camera::DisplayLastFrame()
{
    if (buffer_frame.GetImg().empty()) 
    {
        SPDLOG_WARN("CAM{}: No frame to display", cam_id);
        return;
    }
    cv::imshow("Camera " + std::to_string(cam_id), buffer_frame.GetImg());
    cv::waitKey(1);
}


void Camera::RunCaptureLoop()
{





}

void Camera::StopCaptureLoop()
{

}