#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/opencv.hpp>


class Camera
{


public:
    Camera(int cam_id);


    void TurnOn();
    void TurnOff();

    void CaptureFrame();


    bool LoadIntrinsics(const cv::Mat& intrinsics, const cv::Mat& distortion_parameters);


private:

    int cam_id;
    cv::VideoCapture cap;
    bool is_camera_on;

    cv::Mat buffer_frame;

    cv::Mat intrinsics;
    cv::Mat distortion_parameters; 


};










#endif // CAMERA_HPP