#include "spdlog/spdlog.h"
#include "frame.hpp"


Frame::Frame(int cam_id, const cv::Mat& img, const std::string& timestamp)
:
cam_id(cam_id),
img(img),
timestamp(timestamp)
{}


int Frame::GetCamId() const
{
    return cam_id;
}


const cv::Mat& Frame::GetImg() const
{
    return img;
}


const std::string& Frame::GetTimestamp() const
{
    return timestamp;
}