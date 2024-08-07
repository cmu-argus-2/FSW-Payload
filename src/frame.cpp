#include "spdlog/spdlog.h"
#include "frame.hpp"


Frame::Frame(int cam_id, const cv::Mat& img, std::int64_t timestamp)
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


const std::int64_t& Frame::GetTimestamp() const
{
    return timestamp;
}