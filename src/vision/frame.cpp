#include "spdlog/spdlog.h"
#include "vision/frame.hpp"

Frame::Frame(int cam_id, const cv::Mat& img, std::int64_t timestamp)
:
_cam_id(cam_id),
_img(img),
_timestamp(timestamp)
{}


Frame::Frame()  
:
_cam_id(-1),
_img(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, CV_8UC3),
_timestamp(0)
{}


void Frame::Update(int cam_id, const cv::Mat& img, std::int64_t timestamp) 
{
    _cam_id = cam_id;
    _img.copyTo(img); // Copy into preallocated buffer
    _timestamp = timestamp;
}


int Frame::GetID() const
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