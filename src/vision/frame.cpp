#include "spdlog/spdlog.h"
#include "vision/frame.hpp"
#include <opencv2/opencv.hpp>

Frame::Frame()  
:
_cam_id(-1),
_img(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, CV_8UC3),
_timestamp(0),
_img_mtx(std::make_shared<std::mutex>())
{}


Frame::Frame(int cam_id, const cv::Mat& img, std::uint64_t timestamp)
    : _cam_id(cam_id),
      _img(img),  
      _timestamp(timestamp),
      _img_mtx(std::make_shared<std::mutex>())
{}

// For rvalue reference
Frame::Frame(int cam_id, cv::Mat&& img, std::uint64_t timestamp)
:
_cam_id(cam_id),
_img(std::move(img)),
_timestamp(timestamp),
_img_mtx(std::make_shared<std::mutex>())
{}

Frame::Frame(const Frame& other)
    : _cam_id(other._cam_id),
      _img(other._img.clone()), 
      _timestamp(other._timestamp),
      _img_mtx(std::make_shared<std::mutex>())  // creating a new mutex per instance to avoid intercopy locking
{}


Frame& Frame::operator=(const Frame& other)
{
    if (this != &other) {
        _img = other._img.clone(); 
        _cam_id = other._cam_id;
        _timestamp = other._timestamp;
        _region_ids = other._region_ids;
        _landmarks = other._landmarks;
        _img_mtx = std::make_shared<std::mutex>(); // new mutex per instance
    }
    return *this;
}

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

bool Frame::IsBlurred()
{
    // the more an image is blurred, the less edges there is 
    // need to lock in there, since the buffer is shared 

    static std::vector<uint8_t> grey_buffer(FULL_RES_PIXEL_SIZE);
    static std::vector<double> laplacian_buffer(FULL_RES_PIXEL_SIZE);

    cv::Scalar mean, std_dev;
    {   
        std::lock_guard<std::mutex> lck(*_img_mtx);
        cv::Size size = _img.size(); 
        cv::Mat gray(size, CV_8U, grey_buffer.data());
        cv::Mat laplacian(size, CV_64F, laplacian_buffer.data());

        cv::cvtColor(_img, gray, cv::COLOR_BGR2GRAY);
        cv::Laplacian(gray, laplacian, CV_64F);

        cv::meanStdDev(laplacian, mean, std_dev);
    }

    double variance = std_dev[0] * std_dev[0];  // variance = sigma^2

    bool is_blurred = variance < BLUR_THRESHOLD;

    SPDLOG_INFO("Variance is {}", variance);
    SPDLOG_INFO("Frame is{}blurred", is_blurred ? " " : " not ");
 
    return is_blurred;

}