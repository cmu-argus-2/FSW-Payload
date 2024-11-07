#include "spdlog/spdlog.h"
#include "vision/frame.hpp"


Frame::Frame(int cam_id, const cv::Mat& img, std::int64_t timestamp)
:
_cam_id(cam_id),
_img(img),
_timestamp(timestamp)
{}


int Frame::GetCamId() const
{
    return _cam_id;
}


const cv::Mat& Frame::GetImg() const
{
    return _img;
}


const std::int64_t& Frame::GetTimestamp() const
{
    return _timestamp;
}