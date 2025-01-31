#ifndef FRAME_HPP
#define FRAME_HPP

#include "spdlog/spdlog.h"
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vision/rc.hpp>
#include <vision/ld.hpp>

// 4608×2592
#define DEFAULT_FRAME_WIDTH 4608
#define DEFAULT_FRAME_HEIGHT 2592
#define EARTH_THRESHOLD 0.7


class Frame 
{
public:
    Frame();
    Frame(int cam_id, const cv::Mat& img, std::uint64_t timestamp);
    Frame(int cam_id, cv::Mat&& img, std::uint64_t timestamp);

    void Update(int cam_id, const cv::Mat& img, std::uint64_t timestamp);
    
    int GetCamID() const;
    const cv::Mat& GetImg() const;
    std::uint64_t GetTimestamp() const;

    const std::vector<RegionID>& GetRegionIDs() const;
    const std::vector<Landmark>& GetLandmarks() const;
    
    void AddRegion(RegionID region_id);
    void ClearRegions();
    void AddLandmark(float x, float y, uint16_t class_id, RegionID region_id);
    void ClearLandmarks();

private:
    int _cam_id;
    cv::Mat _img;
    std::uint64_t _timestamp;

    std::vector<RegionID> _region_ids;  // Container for regions
    std::vector<Landmark> _landmarks;  // Container for landmarks
};






#endif // FRAME_HPP