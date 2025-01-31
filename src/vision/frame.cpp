#include "spdlog/spdlog.h"
#include "vision/frame.hpp"

Frame::Frame()  
:
_cam_id(-1),
_img(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, CV_8UC3),
_timestamp(0)
{}


Frame::Frame(int cam_id, const cv::Mat& img, std::uint64_t timestamp)
    : _cam_id(cam_id),
      _img(img),  
      _timestamp(timestamp)
{}

// For rvalue reference
Frame::Frame(int cam_id, cv::Mat&& img, std::uint64_t timestamp)
:
_cam_id(cam_id),
_img(std::move(img)),
_timestamp(timestamp)
{}


void Frame::Update(int cam_id, const cv::Mat& img, std::uint64_t timestamp) 
{
    _cam_id = cam_id;
    img.copyTo(_img); // Copy into preallocated buffer
    _timestamp = timestamp;
}


int Frame::GetCamID() const
{
    return _cam_id;
}


const cv::Mat& Frame::GetImg() const
{
    return _img;
}


std::uint64_t Frame::GetTimestamp() const
{
    return _timestamp;
}


const std::vector<RegionID>& Frame::GetRegionIDs() const
{
    return _region_ids;
}

const std::vector<Landmark>& Frame::GetLandmarks() const
{
    return _landmarks;
}

void Frame::AddRegion(RegionID region_id)
{
    _region_ids.push_back(region_id);
}


void Frame::AddLandmark(float x, float y, uint16_t class_id, RegionID region_id)
{
    _landmarks.emplace_back(x, y, class_id, region_id);
}

void Frame::ClearLandmarks()
{
    _landmarks.clear();
}