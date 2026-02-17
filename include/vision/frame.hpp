#ifndef FRAME_HPP
#define FRAME_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <tuple>

#include <opencv2/opencv.hpp>

#include <vision/regions.hpp>
#include <vision/ld.hpp>
#include <nlohmann/json.hpp>
using Json = nlohmann::json;


// 4608×2592
inline constexpr int DEFAULT_FRAME_WIDTH = 4608;
inline constexpr int DEFAULT_FRAME_HEIGHT = 2592;
inline constexpr uint64_t FULL_RES_PIXEL_SIZE = DEFAULT_FRAME_WIDTH * DEFAULT_FRAME_HEIGHT;

inline constexpr double EARTH_THRESHOLD = 0.7;
inline constexpr double BLUR_THRESHOLD = 50.0;

enum class ImageState : uint8_t {
    NotEarth    = 0,
    Earth       = 1,
    HasRegion   = 2,
    HasLandmark = 3
};

// Processing stage for frame pipeline
enum class ProcessingStage : uint8_t {
    NotPrefiltered = 0, // 0 = not pre-filtered
    Prefiltered    = 1, // 1 = pre-filtered
    RCNeted        = 2, // 2 = region classification done
    LDNeted        = 3  // 3 = landmark detection done
};

inline constexpr ProcessingStage ToProcessingStage(uint8_t v) {
    switch (v) {
        case 0: return ProcessingStage::NotPrefiltered;
        case 1: return ProcessingStage::Prefiltered;
        case 2: return ProcessingStage::RCNeted;
        default: return ProcessingStage::LDNeted;
    }
}

inline constexpr uint8_t ProcessingStageIndex(ProcessingStage s) {
    return static_cast<uint8_t>(s);
}

inline constexpr ProcessingStage NextStage(ProcessingStage s) {
    return (s == ProcessingStage::LDNeted)
        ? s
        : static_cast<ProcessingStage>(static_cast<uint8_t>(s) + 1u);
}

inline constexpr bool IsAtLeast(ProcessingStage current, ProcessingStage required) {
    return static_cast<uint8_t>(current) >= static_cast<uint8_t>(required);
}

inline const char* ProcessingStageToString(ProcessingStage s) {
    switch (s) {
        case ProcessingStage::NotPrefiltered: return "NotPrefiltered";
        case ProcessingStage::Prefiltered:    return "Prefiltered";
        case ProcessingStage::RCNeted:        return "RCneted";
        case ProcessingStage::LDNeted:        return "LDNeted";
        default:                              return "Unknown";
    }
}


class Frame 
{
public:
    Frame();
    Frame(int cam_id, const cv::Mat& img, std::uint64_t timestamp);
    Frame(int cam_id, cv::Mat&& img, std::uint64_t timestamp);
    Frame(const Frame& other);

    Frame& operator=(const Frame& other);

    bool operator>(const Frame& other) const;
    bool operator<(const Frame& other) const;
    bool operator>=(const Frame& other) const;
    bool operator<=(const Frame& other) const;
    bool operator==(const Frame& other) const;

    void Update(int cam_id, const cv::Mat& img, std::uint64_t timestamp);
    
    int GetCamID() const;
    const cv::Mat& GetImg() const;
    std::uint64_t GetTimestamp() const;
    const ImageState GetImageState() const;
    const ProcessingStage GetProcessingStage() const;
    const float GetRank() const;
    Json toJson() const;
    nlohmann::ordered_json toOrderedJson() const;
    void fromJson(const Json& j);


    const std::vector<RegionID>& GetRegionIDs() const;
    const std::vector<float>& GetRegionConfidences() const;
    const std::vector<Landmark>& GetLandmarks() const;
    
    void AddRegion(RegionID region_id, float confidence);
    void ClearRegions();
    void AddLandmark(const Landmark& landmark); 
    void AddLandmark(float x, float y, uint16_t class_id, RegionID region_id, float height_, float width_, float confidence_);
    void ClearLandmarks();
    void RunPrefiltering();

    bool IsBlurred();

private:
    int _cam_id;
    cv::Mat _img;
    std::uint64_t _timestamp;

    ImageState _annotation_state;
    float _rank; // score to rank images with the same annotation_state (higher = better)
    ProcessingStage _processing_stage;
    std::vector<RegionID> _region_ids;  // Container for regions
    std::vector<float> _region_confidences; // Confidence values aligned with _region_ids
    std::vector<Landmark> _landmarks;  // Container for landmarks

    // mutex is not copyable
    std::shared_ptr<std::mutex> _img_mtx; // using shared_ptr for copyability (private anyway)
};


// Helper functions for parsing the output of RC net and LD net
namespace VisionJson
{

using Json = nlohmann::json;

inline std::vector<std::string> RegionIdsToStrings(const std::vector<RegionID>& region_ids)
{
    std::vector<std::string> region_codes;
    region_codes.reserve(region_ids.size());
    for (const auto& region_id : region_ids)
    {
        region_codes.emplace_back(GetRegionString(region_id));
    }
    return region_codes;
}

inline void LoadRegionIdsFromJson(const Json& region_json, std::vector<RegionID>& region_ids)
{
    region_ids.clear();
    if (!region_json.is_array())
    {
        return;
    }

    for (const auto& region_item : region_json)
    {
        if (region_item.is_string())
        {
            RegionID region_id = GetRegionID(region_item.get<std::string>());
            if (region_id != RegionID::UNKNOWN)
            {
                region_ids.push_back(region_id);
            }
        }
    }
}

} // namespace VisionJson




#endif // FRAME_HPP