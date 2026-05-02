#include "spdlog/spdlog.h"
#include "vision/frame.hpp"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <numeric>
using Json = nlohmann::json;


Frame::Frame()
    :
    _cam_id(-1),
    _img(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT, CV_8UC3),
    _timestamp(0),
    _img_mtx(std::make_shared<std::mutex>()),
    _annotation_state(ImageState::NotEarth),
    _rank(static_cast<float>(ImageState::NotEarth)),
    _processing_stage(ProcessingStage::NotPrefiltered)
{}


Frame::Frame(int cam_id, const cv::Mat& img, std::uint64_t timestamp)
    : _cam_id(cam_id),
      _img(img),
      _timestamp(timestamp),
      _img_mtx(std::make_shared<std::mutex>()),
      _annotation_state(ImageState::NotEarth),
      _rank(static_cast<float>(ImageState::NotEarth)),
      _processing_stage(ProcessingStage::NotPrefiltered)
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
_processing_stage(ProcessingStage::NotPrefiltered)
{}

Frame::Frame(const Frame& other)
    : _cam_id(other._cam_id),
      _img(other._img.clone()),
      _timestamp(other._timestamp),
      _img_mtx(std::make_shared<std::mutex>()),  // creating a new mutex per instance to avoid intercopy locking
      _annotation_state(other._annotation_state),
      _rank(other._rank),
      _processing_stage(other._processing_stage),
      _inference_results(other._inference_results),
      _prefilter_result(other._prefilter_result)
{}


Frame& Frame::operator=(const Frame& other)
{
    if (this != &other) {
        _img = other._img.clone();
        _cam_id = other._cam_id;
        _timestamp = other._timestamp;
        _inference_results = other._inference_results;
        _prefilter_result = other._prefilter_result;
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

const std::optional<InferenceResults>& Frame::GetInferenceResults() const
{
    return _inference_results;
}

void Frame::SetInferenceResults(InferenceResults results)
{
    _inference_results = std::move(results);
    UpdateAnnotationState();
    UpdateRank();
}

void Frame::ClearInferenceResults()
{
    _inference_results.reset();
    UpdateAnnotationState();
    UpdateRank();
}

const std::vector<Region>& Frame::GetRegions() const
{
    static const std::vector<Region> empty{};
    return _inference_results.has_value() ? _inference_results->regions : empty;
}

const std::vector<RegionID> Frame::GetRegionIDs() const
{
    if (!_inference_results.has_value()) return {};
    std::vector<RegionID> ids;
    for (const auto& r : _inference_results->regions) ids.push_back(r.id);
    return ids;
}

const std::vector<float> Frame::GetRegionConfidences() const
{
    if (!_inference_results.has_value()) return {};
    std::vector<float> confs;
    for (const auto& r : _inference_results->regions) confs.push_back(r.confidence);
    return confs;
}

const std::vector<Landmark>& Frame::GetLandmarks() const
{
    static const std::vector<Landmark> empty{};
    return _inference_results.has_value() ? _inference_results->landmarks : empty;
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

const std::optional<PrefilterResult>& Frame::GetPrefilterResult() const
{
    return _prefilter_result;
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
        // Prefiltering results (omitted if not yet run)
        if (_prefilter_result.has_value()) {
            j["prefilter"] = PrefilterResultToJson(*_prefilter_result);
        }
        // 2. Inference results (omitted if inference has not been run)
        if (_inference_results.has_value())
            j["inference_results"] = _inference_results->toJson();
        
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
        // Prefiltering results
        _prefilter_result.reset();
        if (j.contains("prefilter") && j.at("prefilter").is_object())
        {
            _prefilter_result = PrefilterResultFromJson(j.at("prefilter"));
        }
        // 2. Inference results — nested under "inference_results";
        //    fall back to flat top-level keys for frames written before this format.
        _inference_results.reset();
        {
            const bool has_nested = j.contains("inference_results") && j.at("inference_results").is_object();
            const Json& inf = has_nested ? j.at("inference_results") : j;
            InferenceResults ir = InferenceResults::fromJson(inf);
            if (ir.rc_version != -1 || !ir.regions.empty() || !ir.landmarks.empty())
                _inference_results = std::move(ir);
        }
    }
    catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to parse Frame from JSON: {}", e.what());
    }
}

void Frame::RunPrefiltering()
{
    if (_processing_stage != ProcessingStage::NotPrefiltered)
    {
        SPDLOG_WARN("Frame has already been pre-filtered. Skipping pre-filtering step.");
        return;
    }
    _prefilter_result = prefilter_image(_img);
    _processing_stage = ProcessingStage::Prefiltered;
    UpdateAnnotationState();
    UpdateRank();
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

void Frame::UpdateAnnotationState()
{
    const bool has_landmarks = _inference_results.has_value() && !_inference_results->landmarks.empty();
    const bool has_regions   = _inference_results.has_value() && !_inference_results->regions.empty();

    if (has_landmarks) {
        _annotation_state = ImageState::HasLandmark;
    } else if (has_regions) {
        _annotation_state = ImageState::HasRegion;
    } else if (_prefilter_result.has_value() && _prefilter_result->passed) {
        _annotation_state = ImageState::Earth;
    } else {
        _annotation_state = ImageState::NotEarth;
    }
}

void Frame::UpdateRank()
{
    switch (_annotation_state) {
        case ImageState::NotEarth:
            _rank = 0.0f;
            break;

        case ImageState::Earth:
            // Rank by cloudiness if prefilter result is available
            if (_prefilter_result.has_value()) {
                _rank = 1.0f - (_prefilter_result->cloudiness / 100.0f);
            } else {
                _rank = 0.5f;
            }
            break;

        case ImageState::HasRegion: {
            if (!_inference_results.has_value()) { _rank = 0.0f; break; }
            const auto& regions = _inference_results->regions;
            float sum = 0.0f;
            for (const auto& r : regions) sum += r.confidence;
            _rank = regions.empty() ? 0.0f : sum / static_cast<float>(regions.size());
            break;
        }

        case ImageState::HasLandmark: {
            if (!_inference_results.has_value()) { _rank = 0.0f; break; }
            float sum = 0.0f;
            for (const auto& l : _inference_results->landmarks) sum += l.confidence;
            _rank = sum / 100.0f;
            break;
        }
    }
}