#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/opencv.hpp>


class Camera
{


public:
    Camera();


    void TurnOn();


    void TurnOff();


private:

    int cam_id;
    cv::VideoCapture cap;
    bool is_camera_on;

    cv::Mat intrinsics;
    cv::Mat distortion_parameters; 


};










#endif // CAMERA_HPP