#include "spdlog/spdlog.h"
#include "vision/frame.hpp"
#include "vision/prefiltering.hpp"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
using Json = nlohmann::json;


Frame::Frame()  
    :
    _cam_id(-1),
    _img(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, CV_8UC3),
    _timestamp(0),
    _img_mtx(std::make_shared<std::mutex>()),
    _annotation_state(ImageState::NotEarth),
    _rank(static_cast<float>(ImageState::NotEarth)),
    _processing_stage(ProcessingStage::NotPrefiltered),
    _regions({}),
    _landmarks({})
{}


Frame::Frame(int cam_id, const cv::Mat& img, std::uint64_t timestamp)
    : _cam_id(cam_id),
      _img(img),  
      _timestamp(timestamp),
      _img_mtx(std::make_shared<std::mutex>()),
      _annotation_state(ImageState::NotEarth),
    _rank(static_cast<float>(ImageState::NotEarth)),
      _processing_stage(ProcessingStage::NotPrefiltered),
      _regions({}),
      _landmarks({})
{}

// For rvalue reference
Frame::Frame(int cam_id, cv::Mat&& img, std::uint64_t timestamp)
:
_cam_id(cam_id),
_img(std::move(img)),
_timestamp(timestamp),
_img_mtx(std::make_shared<std::mutex>()),
_annotation_state(ImageState::NotEarth),
_rank(static_cast<float>(ImageState::NotEarth)),
_processing_stage(ProcessingStage::NotPrefiltered),
_regions({}),
_landmarks({})
{}

Frame::Frame(const Frame& other)
    : _cam_id(other._cam_id),
      _img(other._img.clone()), 
      _timestamp(other._timestamp),
      _img_mtx(std::make_shared<std::mutex>()),  // creating a new mutex per instance to avoid intercopy locking
      _annotation_state(other._annotation_state),
      _rank(other._rank),
      _processing_stage(other._processing_stage),
      _regions(other._regions),
      _landmarks(other._landmarks)
{}


Frame& Frame::operator=(const Frame& other)
{
    if (this != &other) {
        _img = other._img.clone(); 
        _cam_id = other._cam_id;
        _timestamp = other._timestamp;
        _regions = other._regions;
        _landmarks = other._landmarks;
        _img_mtx = std::make_shared<std::mutex>(); // new mutex per instance
        _annotation_state = other._annotation_state;
        _rank = other._rank;
        _processing_stage = other._processing_stage;
    }
    return *this;
}

bool Frame::operator>(const Frame& other) const
{
    return std::tie(this->_annotation_state, this->_rank, this->_timestamp) > std::tie(other._annotation_state, other._rank, other._timestamp);
}
bool Frame::operator>=(const Frame& other) const
{
    return std::tie(this->_annotation_state, this->_rank, this->_timestamp) >= std::tie(other._annotation_state, other._rank, other._timestamp);
}
bool Frame::operator<(const Frame& other) const
{
    return std::tie(this->_annotation_state, this->_rank, this->_timestamp) < std::tie(other._annotation_state, other._rank, other._timestamp);
}
bool Frame::operator<=(const Frame& other) const
{
    return std::tie(this->_annotation_state, this->_rank, this->_timestamp) <= std::tie(other._annotation_state, other._rank, other._timestamp);
}
// May want an == operator to check if the images are the same instead, in case 
// one somehow gets duplicated. That should never happen though
bool Frame::operator==(const Frame& other) const
{
    return this->_annotation_state == other._annotation_state && this->_rank == other._rank && this->_timestamp == other._timestamp;
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

const std::vector<Region>& Frame::GetRegions() const
{
    return _regions;
}

const std::vector<RegionID> Frame::GetRegionIDs() const
{
    std::vector<RegionID> region_ids;
    for (const auto& region : _regions)
    {
        region_ids.push_back(region.id);
    }
    return region_ids;
}

const std::vector<float> Frame::GetRegionConfidences() const
{
    std::vector<float> region_confidences;
    for (const auto& region : _regions)
    {
        region_confidences.push_back(region.confidence);
    }
    return region_confidences;
}

const std::vector<Landmark>& Frame::GetLandmarks() const
{
    return _landmarks;
}

const ImageState Frame::GetImageState() const
{
    return _annotation_state;
}

const ProcessingStage Frame::GetProcessingStage() const
{
    return _processing_stage;
}

void Frame::SetProcessingStage(ProcessingStage stage)
{
    _processing_stage = stage;
}

const float Frame::GetRank() const
{
    return _rank;
}

Json Frame::toJson() const
{
    return Json(toOrderedJson());
}

nlohmann::ordered_json Frame::toOrderedJson() const // Order the Json keys
{
    nlohmann::ordered_json j;
    try {
        // 1. Image header information
        j["timestamp"] = _timestamp;
        j["cam_id"] = _cam_id;
        j["annotation_state"] = static_cast<int>(_annotation_state);
        j["processing_stage"] = static_cast<int>(_processing_stage);
        j["rank"] = _rank;
        j["detected_regions_count"] = _regions.size();
        j["detected_landmarks_count"] = _landmarks.size();
        // Possibly add more information from the prefiltering results to the header
        // 2. Inference results
        // 2.1. List of regions
        j["regions"] = nlohmann::ordered_json::array();
        for (size_t i = 0; i < _regions.size(); ++i)
        {            
            const auto& region = _regions[i];
            nlohmann::ordered_json region_json;
            region_json["region_" + std::to_string(i)] = {
                {"id", region.id},
                {"confidence", region.confidence}
            };
            j["regions"].push_back(region_json);
        }
        // 2.2. List of landmarks
        j["landmarks"] = nlohmann::ordered_json::array();
        for (size_t i = 0; i < _landmarks.size(); ++i)
        {            
            const auto& landmark = _landmarks[i];
            nlohmann::ordered_json landmark_json;
            landmark_json["landmark_" + std::to_string(i)] = {
                {"x", landmark.x},
                {"y", landmark.y},
                {"height", landmark.height},
                {"width", landmark.width},
                {"confidence", landmark.confidence},
                {"class_id", landmark.class_id},
                {"region_id", landmark.region_id}
            };
            j["landmarks"].push_back(landmark_json);
        }
        
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to convert Frame to JSON: {}", e.what());
    }
    return j;
}

void Frame::fromJson(const Json& j) // Write values to JSON
{
    try {
        // 1. Image header information
        _timestamp = j.at("timestamp").get<std::uint64_t>();
        _cam_id = j.at("cam_id").get<int>();
        _annotation_state = static_cast<ImageState>(j.at("annotation_state").get<int>());
        _processing_stage = static_cast<ProcessingStage>(j.at("processing_stage").get<int>());
        _rank = j.at("rank").get<float>();
        int detected_regions_count = j.at("detected_regions_count").get<int>();
        int detected_landmarks_count = j.at("detected_landmarks_count").get<int>();
        // 2. Inference results
        // 2.1. List of regions
        _regions.clear();
        if (j.contains("regions") && j.at("regions").is_array())
        {
            for (const auto& region_item : j.at("regions"))
            {
                for (const auto& [key, value] : region_item.items())
                {
                    RegionID region_id = static_cast<RegionID>(value.at("id").get<int>());
                    float confidence = value.at("confidence").get<float>();
                    _regions.emplace_back(region_id, confidence);
                }
            }
        }
        // 2.2. List of landmarks
        _landmarks.clear();
        if (j.contains("landmarks") && j.at("landmarks").is_array())
        {
            for (const auto& landmark_item : j.at("landmarks"))
            {
                for (const auto& [key, value] : landmark_item.items())
                {
                    float x = value.at("x").get<float>();
                    float y = value.at("y").get<float>();
                    float height = value.at("height").get<float>();
                    float width = value.at("width").get<float>();
                    float confidence = value.at("confidence").get<float>();
                    uint16_t class_id = value.at("class_id").get<uint16_t>();
                    RegionID region_id = static_cast<RegionID>(value.at("region_id").get<int>());
                    _landmarks.emplace_back(x, y, class_id, region_id, height, width, confidence);
                }
            }
        }
    }
    catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to parse Frame from JSON: {}", e.what());
    }
}

void Frame::AddRegion(RegionID region_id, float confidence)
{
    _regions.emplace_back(region_id, confidence);
    if (_rank < static_cast<float>(ImageState::HasRegion))
    {
        _rank = 2.0f; // TODO: make a rank regions function for the frame
        // Something like average or max region confidence
    }
    if (_annotation_state < ImageState::HasRegion)
    {
        _annotation_state = ImageState::HasRegion; // if it has a region, it must be earth
    }
}

void Frame::ClearRegions()
{
    _regions.clear();
    // TODO: potentially readjust rank and annotation state here
}

void Frame::AddLandmark(const Landmark& landmark)
{
    if (_rank < static_cast<float>(ImageState::HasLandmark))
    {
        _rank = 3.0f; // TODO: make a rank landmarks function for the frame.
        // Maybe weighted sum of confidence and region confidence?
    }
    if (_annotation_state < ImageState::HasLandmark)
    {
        _annotation_state = ImageState::HasLandmark;
    }
    _landmarks.push_back(landmark);
}

void Frame::AddLandmark(float x, float y, uint16_t class_id, RegionID region_id, float height_, float width_, float confidence_)
{
    if (_rank < static_cast<float>(ImageState::HasLandmark))
    {
        _rank = 3.0f; // need to rank images with landmarks. Maybe weighted sum of confidence and region confidence?
    }
    if (_annotation_state < ImageState::HasLandmark)
    {
        _annotation_state = ImageState::HasLandmark;
    }
    _landmarks.emplace_back(x, y, class_id, region_id, height_, width_, confidence_);
}

void Frame::ClearLandmarks()
{
    _landmarks.clear();
    // TODO: potentially readjust rank and annotation state here
}

void Frame::RunPrefiltering()
{
    if (_processing_stage != ProcessingStage::NotPrefiltered)
    {
        SPDLOG_WARN("Frame has already been pre-filtered. Skipping pre-filtering step.");
        return;
    }
    PrefilterResult res = prefilter_image(_img);
    if (res.passed)
    {
        _annotation_state = ImageState::Earth;
        _rank = 1.0f - (res.cloudiness / 100.0f); // higher cloudiness = lower rank
    }
    else
    {
        _annotation_state = ImageState::NotEarth;
        _rank = static_cast<float>(ImageState::NotEarth);
    }
    
    _processing_stage = ProcessingStage::Prefiltered;
}

bool Frame::IsBlurred()
{
    // the more an image is blurred, the less edges there are
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