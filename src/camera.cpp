#include "spdlog/spdlog.h"
#include <cstdint>
#include <chrono>
#include "camera.hpp"


Camera::Camera(int cam_id)
: 
cam_id(cam_id),
is_camera_on(false),
buffer_frame(cam_id, cv::Mat(), 0),
display_flag(false),
loop_flag(false)
{
}


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

void Camera::CaptureFrame()
{
    
    cv::Mat captured_frame;
    try {
        if (is_camera_on) 
        {   
            std::int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            cap >> captured_frame;

            if (captured_frame.empty()) {
                SPDLOG_ERROR("Unable to capture frame");
                throw std::runtime_error("Unable to capture frame");
            }
            
            buffer_frame = Frame(cam_id, captured_frame, timestamp);
            SPDLOG_INFO("CAM{}: Frame captured successfully at {}", cam_id, timestamp);

        }
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Exception occurred: ", e.what());
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

void Camera::RunLoop()
{
    
    // Should avoid while loop
    while (loop_flag)
    {

        if (is_camera_on) 
        {
            CaptureFrame();
        }

        if (display_flag && !buffer_frame.GetImg().empty()) 
        {
            cv::imshow("CAM" + std::to_string(cam_id), buffer_frame.GetImg());
            cv::waitKey(1);
        }

    }


}


void Camera::StopLoop()
{
    this->loop_flag = false;
}


void Camera::DisplayLoop(bool display_flag)
{
    // std::lock_guard<std::mutex> lock(display_mutex); // disappears when out of scope
    this->display_flag = display_flag;
}