#include <cstdint>
#include <chrono>
#include "spdlog/spdlog.h"
#include "vision/camera.hpp"
#include "core/timing.hpp"

Camera::Camera(int cam_id, std::string path)
: 
width(DEFAULT_FRAME_WIDTH),
height(DEFAULT_FRAME_HEIGHT),
cam_status(CAM_STATUS::INACTIVE),
last_error(CAM_ERROR::NO_ERROR),
consecutive_error_count(0),
cam_id(cam_id),
cam_path(path),
local_buffer_img(cv::Mat(height, width, CV_8UC3)),
buffer_frame(cam_id, cv::Mat(height, width, CV_8UC3), 0),
_new_frame_flag(false)
{
}



bool Camera::Enable()
{
    
    if (cam_status == CAM_STATUS::ACTIVE) {
        SPDLOG_WARN("CAM{}: Camera is already active", cam_id);
        return true;
    }

    bool success = false;

    try 
    {

        // Try opening with default backend first 
        cap.open(cam_path);
        
        // If that fails, try with explicit V4L2 backend
        if (!cap.isOpened()) {
            SPDLOG_WARN("CAM{}: Default backend failed, trying V4L2", cam_id);
            cap.open(cam_path, cv::CAP_V4L2);
        }

        // Check if the camera is opened successfully
        if (!cap.isOpened()) {
            SPDLOG_WARN("Unable to open the camera");
            throw std::runtime_error("Unable to open the camera");
        }

        // Set the camera resolution - remove by reading confif file
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap.set(cv::CAP_PROP_FPS, DEFAULT_CAMERA_FPS);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 3); 
        // cap.set(cv::CAP_PROP_GAIN, 10000); // Set gain to maximum - OpenCV will clamp it to the maximum value
        SPDLOG_INFO("CAM{}: Camera gain set to {}", cam_id, cap.get(cv::CAP_PROP_GAIN));

        // Warm up camera by reading several frames 
        cv::Mat dummy_frame;
        for (int i = 0; i < 10; i++) {
            cap >> dummy_frame;
        }
        SPDLOG_INFO("CAM{}: Camera warmup complete", cam_id);

        // Start capture loop
        cam_status = CAM_STATUS::ACTIVE;
        capture_thread = std::thread(&Camera::RunCaptureLoop, this);
        SPDLOG_INFO("CAM{}: Camera successfully turned on", cam_id);

        success = true;


    } 
    catch (const std::exception& e) 
    {
        SPDLOG_WARN("Exception occurred: {}", e.what());


        HandleErrors(CAM_ERROR::INITIALIZATION_FAILED);

        // only be necessary if the camera was opened but failed afterwards
        if (cap.isOpened()) {
            cap.release();
        }

    }

    return success;
}

bool Camera::Disable()
{
    StopCaptureLoop();

    switch (cam_status)
    {

        case CAM_STATUS::INACTIVE:
        {
            SPDLOG_WARN("CAM{}: Camera is already inactive", cam_id);
            cap.release();
            return true;
        }

        case CAM_STATUS::ACTIVE:
        {
            StopCaptureLoop();
            cap.release();
            cam_status = CAM_STATUS::INACTIVE;
            SPDLOG_INFO("CAM{}: Camera successfully disabled.", cam_id);
            return true;
        }

        default:
        {
            SPDLOG_ERROR("CAM{}: Unknown camera status", cam_id);
            // TODO: Handle error
            return false;
        }

    }

}

void Camera::CaptureFrame()
{
    // Single responsibility principle. Status must be checked externally

    // static cv::Mat local_buffer_img(height, width, CV_8UC3); // pre-allocate memory
    
    // Capture frame without holding the mutex - TODO access the time ref from the hardware API 
    auto timestamp = timing::GetCurrentTimeMs();

    // SPDLOG_CRITICAL("CAM{}: copying frame", cam_id);
    cap >> local_buffer_img;
    // SPDLOG_CRITICAL("CAM{}: frame copied", cam_id);

    if (local_buffer_img.empty()) 
    {
        std::lock_guard<std::shared_mutex> lock(frame_mutex);
        _new_frame_flag.store(false);
        // local_buffer_img = cv::Mat::zeros(height, width, CV_8UC3);
        SPDLOG_ERROR("Unable to capture frame");
        HandleErrors(CAM_ERROR::CAPTURE_FAILED);
        return;
    }

    // Lock only for modifying shared resources
    {
        std::lock_guard<std::shared_mutex> lock(frame_mutex); // Exclusive lock
        buffer_frame.Update(cam_id, local_buffer_img, static_cast<uint64_t>(timestamp));
        _new_frame_flag.store(true);
    }

    HandleErrors(CAM_ERROR::NO_ERROR);
}


void Camera::SetOffNewFrameFlag()
{
    _new_frame_flag.store(false);
}

bool Camera::IsNewFrameAvailable() const
{
    return _new_frame_flag.load();
}

EC Camera::MapError(CAM_ERROR cam_error)
{
    switch (cam_error)
    {
        case CAM_ERROR::NO_ERROR:
            return EC::OK;
        case CAM_ERROR::CAPTURE_FAILED:
            return EC::CAMERA_CAPTURE_FAILED;
        case CAM_ERROR::INITIALIZATION_FAILED:
            return EC::CAMERA_INITIALIZATION_FAILED;  
    }

    return EC::OK;
}

void Camera::HandleErrors(CAM_ERROR error)
{
    last_error = error;
    LogError(MapError(error));

    if (last_error == CAM_ERROR::NO_ERROR) {
        consecutive_error_count = 0;
        return;
    }

    consecutive_error_count++;
    SPDLOG_ERROR("CAM{}: Error occurred: {}", cam_id, static_cast<uint8_t>(last_error));

    /*if (consecutive_error_count >= MAX_CONSECUTIVE_ERROR_COUNT) {
        cam_status = CAM_STATUS::DISABLED;
        SPDLOG_ERROR("CAM{}: Camera disabled after {} errors", cam_id, consecutive_error_count);
        consecutive_error_count = 0; // Reset the count for next retry? // Could be stuck in a loop
        return;
    }*/

    if (last_error == CAM_ERROR::INITIALIZATION_FAILED) {
        cam_status = CAM_STATUS::INACTIVE;
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
    // Shared lock for reading shared resources
    std::shared_lock<std::shared_mutex> lock(frame_mutex);
    return buffer_frame;
}

void Camera::CopyBufferFrame(Frame& dest) const
{
    std::shared_lock<std::shared_mutex> lock(frame_mutex);
    dest.Update(buffer_frame.GetCamID(), buffer_frame.GetImg(), buffer_frame.GetTimestamp());
}


bool Camera::IsEnabled() const
{
    return (cam_status == CAM_STATUS::INACTIVE) ? false : true;
}

int Camera::GetID() const
{
    return cam_id;
}

CAM_STATUS Camera::GetStatus() const
{
    return cam_status;
}

void Camera::DisplayLastFrame()
{
    {
        std::shared_lock<std::shared_mutex> lock(frame_mutex);
        if (buffer_frame.GetImg().empty()) 
        {
            SPDLOG_WARN("CAM{}: No frame to display", cam_id);
            return;
        }
        cv::imshow("Camera " + std::to_string(cam_id), buffer_frame.GetImg());
    }
    cv::waitKey(1);
}


void Camera::RunCaptureLoop()
{
    
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = std::chrono::high_resolution_clock::now();
    capture_loop_flag = true;

    while (cam_status == CAM_STATUS::ACTIVE && capture_loop_flag.load()) 
    {
        t1 = std::chrono::high_resolution_clock::now();
        CaptureFrame();
        t2 = std::chrono::high_resolution_clock::now();
        SPDLOG_DEBUG("Capture time: {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count(););
    }
}

void Camera::StopCaptureLoop()
{
    capture_loop_flag.store(false);
    if (capture_thread.joinable()) {
        capture_thread.join();
    }
}

bool Camera::IsCaptureLoopRunning() const
{
    return capture_loop_flag.load();
}

CAM_ERROR Camera::GetLastError() const
{
    return last_error;
}