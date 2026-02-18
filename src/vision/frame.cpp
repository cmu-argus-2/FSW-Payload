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
    _region_ids({}),
    _region_confidences({}),
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
      _region_ids({}),
      _region_confidences({}),
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
_region_ids({}),
_region_confidences({}),
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
      _region_ids(other._region_ids),
      _region_confidences(other._region_confidences),
      _landmarks(other._landmarks)
{}


Frame& Frame::operator=(const Frame& other)
{
    if (this != &other) {
        _img = other._img.clone(); 
        _cam_id = other._cam_id;
        _timestamp = other._timestamp;
        _region_ids = other._region_ids;
        _region_confidences = other._region_confidences;
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


const std::vector<RegionID>& Frame::GetRegionIDs() const
{
    return _region_ids;
}

const std::vector<float>& Frame::GetRegionConfidences() const
{
    return _region_confidences;
}

const std::vector<Landmark>& Frame::GetLandmarks() const
{
    return _landmarks;
}

std::vector<Frame::RegionLandmarkData> Frame::GetRegionLandmarkData() const
{
    std::vector<Frame::RegionLandmarkData> grouped_data;
    grouped_data.reserve(_region_ids.size());

    for (size_t i = 0; i < _region_ids.size(); ++i)
    {
        Frame::RegionLandmarkData region_data;
        region_data.region_id = _region_ids[i];
        region_data.region_confidence = (i < _region_confidences.size()) ? _region_confidences[i] : 0.0f;

        for (const auto& landmark : _landmarks)
        {
            if (landmark.region_id == region_data.region_id)
            {
                region_data.landmark_ids.push_back(landmark.class_id);
                region_data.landmark_confidences.push_back(landmark.confidence);
            }
        }

        grouped_data.push_back(std::move(region_data));
    }

    return grouped_data;
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
        j["timestamp"] = _timestamp;
        j["cam_id"] = _cam_id;
        j["annotation_state"] = static_cast<int>(_annotation_state);
        j["processing_stage"] = static_cast<int>(_processing_stage);
        j["rank"] = _rank;
        j["regions_detected"] = VisionJson::RegionIdsToStrings(_region_ids);
        j["region_confidences"] = _region_confidences;

        // The landmarks are nested arrays with the same indexing as regions_detected and region_confidences
        nlohmann::ordered_json landmark_ids_by_region = nlohmann::ordered_json::array();
        nlohmann::ordered_json landmark_confidences_by_region = nlohmann::ordered_json::array();
        nlohmann::ordered_json landmarks_detected_by_region = nlohmann::ordered_json::array();
        const auto grouped_data = GetRegionLandmarkData();
        for (const auto& region_data : grouped_data)
        {
            landmarks_detected_by_region.push_back(region_data.landmark_ids.size());
            landmark_ids_by_region.push_back(region_data.landmark_ids);
            landmark_confidences_by_region.push_back(region_data.landmark_confidences);
        }

        j["landmarks_by_region"] = {
            {"landmarks_detected", landmarks_detected_by_region},
            {"landmark_ids", landmark_ids_by_region},
            {"landmark_confidences", landmark_confidences_by_region}
        };
        
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to convert Frame to JSON: {}", e.what());
    }
    return j;
}

void Frame::fromJson(const Json& j) // Write values to JSON
{
    try {
        _timestamp = j.at("timestamp").get<std::uint64_t>();
        _cam_id = j.at("cam_id").get<int>();
        _annotation_state = static_cast<ImageState>(j.at("annotation_state").get<int>());
        _processing_stage = static_cast<ProcessingStage>(j.at("processing_stage").get<int>());
        _rank = j.at("rank").get<float>();

        if (j.contains("regions_detected") && j.at("regions_detected").is_array())
        {
            VisionJson::LoadRegionIdsFromJson(j.at("regions_detected"), _region_ids);
        }
        _region_confidences.clear();
        if (j.contains("region_confidences") && j.at("region_confidences").is_array())
        {
            for (const auto& conf : j.at("region_confidences"))
            {
                _region_confidences.emplace_back(conf.get<float>());
            }
        }

        if (_region_confidences.size() != _region_ids.size())
        {
            _region_confidences.assign(_region_ids.size(), 0.0f);
        }
    }
    catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to parse Frame from JSON: {}", e.what());
    }
}

void Frame::AddRegion(RegionID region_id, float confidence)
{
    _region_ids.push_back(region_id);
    _region_confidences.push_back(confidence);
    _rank = static_cast<float>(ImageState::HasRegion);
}

void Frame::ClearRegions()
{
    _region_ids.clear();
    _region_confidences.clear();
}

void Frame::AddLandmark(const Landmark& landmark)
{
    if (_landmarks.empty())
    {
    _rank = static_cast<float>(ImageState::HasLandmark);
    }
    _landmarks.push_back(landmark);
}

void Frame::AddLandmark(float x, float y, uint16_t class_id, RegionID region_id, float height_, float width_, float confidence_)
{
    if (_landmarks.empty())
    {
    _rank = static_cast<float>(ImageState::HasLandmark);
    }
    _landmarks.emplace_back(x, y, class_id, region_id, height_, width_, confidence_);
}

void Frame::ClearLandmarks()
{
    _landmarks.clear();
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