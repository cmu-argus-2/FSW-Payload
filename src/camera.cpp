#include <cstdint>
#include <chrono>
#include "spdlog/spdlog.h"
#include "camera.hpp"


Camera::Camera(int cam_id, std::string path, bool enabled)
: 
cam_status((enabled == false) ? CAM_STATUS::DISABLED : CAM_STATUS::TURNED_OFF),
last_error(CAM_ERROR::NO_ERROR),
cam_id(cam_id),
cam_path(path),
buffer_frame(cam_id, cv::Mat(), 0)
{
}

Camera::Camera(const CameraConfig& config)
:
cam_status((config.enable == false) ? CAM_STATUS::DISABLED : CAM_STATUS::TURNED_OFF),
last_error(CAM_ERROR::NO_ERROR),
cam_id(config.id),
cam_path(config.path),
buffer_frame(cam_id, cv::Mat(), 0)
{
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

        // Set the camera resolution
        cap.set(cv::CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT);

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

    cv::Mat captured_frame;
    try {
        
        if (cam_status == CAM_STATUS::TURNED_ON) 
        {   
            std::int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

            cap >> captured_frame;

            if (captured_frame.empty()) {
                SPDLOG_ERROR("Unable to capture frame");
                throw std::runtime_error("Unable to capture frame");
            }
            
            buffer_frame = Frame(cam_id, captured_frame, timestamp);
            SPDLOG_INFO("CAM{}: Frame captured successfully at {}", cam_id, timestamp);
            return true;
        }

  

    } catch (const std::exception& e) {
        SPDLOG_ERROR("Exception occurred: ", e.what());
        HandleErrors(CAM_ERROR::CAPTURE_FAILED);
    }

    return false;    
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
        SPDLOG_ERROR("CAM{}: Camera disabled", cam_id);
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

const int Camera::GetCamId() const
{
    return cam_id;
}

CAM_STATUS Camera::GetCamStatus() const
{
    return cam_status;
}



