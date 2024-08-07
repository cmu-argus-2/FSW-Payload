#ifndef FRAME_HPP
#define FRAME_HPP

#include "spdlog/spdlog.h"
#include <opencv2/opencv.hpp>

#include <string>


class Frame 
{

public:

    Frame(int cam_id, const cv::Mat& img, const std::string& timestamp);


    int GetCamId() const;
    const cv::Mat& GetImg() const;
    const std::string& GetTimestamp() const;



private:

    int cam_id;
    cv::Mat img;
    std::string timestamp;

};;







#endif // FRAME_HPP