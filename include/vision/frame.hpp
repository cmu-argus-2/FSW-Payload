#ifndef FRAME_HPP
#define FRAME_HPP

#include "spdlog/spdlog.h"
#include <cstdint>
#include <opencv2/opencv.hpp>

// 4608×2592
#define DEFAULT_FRAME_WIDTH 4608
#define DEFAULT_FRAME_HEIGHT 2592

struct Frame 
{
    int _cam_id;
    cv::Mat _img;
    std::uint64_t _timestamp;
    std::string _region; // Detected region of interest
    // Landmark data structure


    Frame();
    Frame(int cam_id, const cv::Mat& img, std::uint64_t timestamp);
    void Update(int cam_id, const cv::Mat& img, std::uint64_t timestamp);
    int GetID() const;
    const cv::Mat& GetImg() const;
    const std::uint64_t& GetTimestamp() const;
    std::string_view GetRegion() const;
    // Locking? 
};







#endif // FRAME_HPP