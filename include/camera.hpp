#ifndef CAMERA_HPP
#define CAMERA_HPP


#include <mutex>
#include <atomic>
#include "frame.hpp"
#include <opencv2/opencv.hpp>
#include "configuration.hpp"


class Camera
{


public:
    Camera(int id, std::string path, bool enabled = false);
    Camera(const CameraConfig& config);


    void TurnOn();
    void TurnOff();

    void CaptureFrame();
    const Frame& GetBufferFrame() const;

    bool IsEnabled() const;


    void RunLoop();
    void StopLoop();
    void DisplayLoop(bool display_flag);


    void LoadIntrinsics(const cv::Mat& intrinsics, const cv::Mat& distortion_parameters);


private:

    
    bool enabled;
    
    std::string cam_path;
    int cam_id;
    cv::VideoCapture cap;
    bool is_camera_on = false;

    Frame buffer_frame;

    cv::Mat intrinsics;
    cv::Mat distortion_parameters; 




    std::atomic<bool> display_flag = false;
    std::atomic<bool> loop_flag = false;


};










#endif // CAMERA_HPP