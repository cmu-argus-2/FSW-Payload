#ifndef FRAME_HPP
#define FRAME_HPP

#include "spdlog/spdlog.h"
#include <cstdint>
#include <opencv2/opencv.hpp>


struct Frame 
{
    int _cam_id;
    cv::Mat _img;
    std::int64_t _timestamp;


    Frame(int cam_id, const cv::Mat& img, std::int64_t timestamp);
    int GetCamId() const;
    const cv::Mat& GetImg() const;
    const std::int64_t& GetTimestamp() const;
    // Locking? 
};







#endif // FRAME_HPP