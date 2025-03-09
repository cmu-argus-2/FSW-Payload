#ifndef FRAME_HPP
#define FRAME_HPP

#include <cstdint>
#include <vector>
#include <mutex>
#include <memory>

#include <opencv2/opencv.hpp>

#include <vision/rc.hpp>
#include <vision/ld.hpp>

// 4608×2592
inline constexpr int DEFAULT_FRAME_WIDTH = 4608;
inline constexpr int DEFAULT_FRAME_HEIGHT = 2592;
inline constexpr uint64_t FULL_RES_PIXEL_SIZE = DEFAULT_FRAME_WIDTH * DEFAULT_FRAME_HEIGHT;

inline constexpr double EARTH_THRESHOLD = 0.7;
inline constexpr double BLUR_THRESHOLD = 50.0;


class Frame 
{
public:
    Frame();
    Frame(int cam_id, const cv::Mat& img, std::uint64_t timestamp);
    Frame(int cam_id, cv::Mat&& img, std::uint64_t timestamp);
    Frame(const Frame& other);

    Frame& operator=(const Frame& other);

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

    bool IsBlurred();

private:
    int _cam_id;
    cv::Mat _img;
    std::uint64_t _timestamp;

    std::vector<RegionID> _region_ids;  // Container for regions
    std::vector<Landmark> _landmarks;  // Container for landmarks

    // mutex is not copyable
    std::shared_ptr<std::mutex> _img_mtx; // using shared_ptr for copyability (private anyway)
};






#endif // FRAME_HPP