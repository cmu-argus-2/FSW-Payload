#include "spdlog/spdlog.h"

#include "camera.hpp"


Camera::Camera()
: 
is_camera_on(false)
{}


void Camera::TurnOn()
{
    try {

        cap.open(cam_id);
        // Check if the camera is opened successfully
        if (!cap.isOpened()) {
            SPDLOG_ERROR("Unable to open the camera");
            throw std::runtime_error("Unable to open the camera");
        }
        is_camera_on = true;

    } catch (const std::exception& e) {
        SPDLOG_ERROR("Exception occurred: ", e.what());
    }

}

void Camera::TurnOff()
{
    try {
        if (is_camera_on) {
            cap.release();
            is_camera_on = false;
        }
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Exception occurred: ", e.what());
    }
}


bool Camera::LoadIntrinsics(const cv::Mat& intrinsics, const cv::Mat& distortion_parameters)
{
    this->intrinsics = intrinsics.clone();
    this->distortion_parameters = distortion_parameters.clone();
    return true;
}