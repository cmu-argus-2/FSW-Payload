#ifndef CAMERA_HPP
#define CAMERA_HPP


#include <mutex>
#include <atomic>
#include "frame.hpp"
#include <opencv2/opencv.hpp>



class Camera
{


public:
    Camera(int cam_id);


    void TurnOn();
    void TurnOff();

    void CaptureFrame();
    const Frame& GetBufferFrame() const;

    


    void RunLoop();
    void StopLoop();
    void DisplayLoop(bool display_flag);


    void LoadIntrinsics(const cv::Mat& intrinsics, const cv::Mat& distortion_parameters);


private:

    int cam_id;
    cv::VideoCapture cap;
    bool is_camera_on;

    Frame buffer_frame;

    cv::Mat intrinsics;
    cv::Mat distortion_parameters; 

    std::atomic<bool> display_flag;
    std::atomic<bool> loop_flag;


};










#endif // CAMERA_HPP