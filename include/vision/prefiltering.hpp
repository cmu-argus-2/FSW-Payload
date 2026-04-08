#ifndef PREFILTERING_H
#define PREFILTERING_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
// using namespace cv;

struct PrefilterResult {
    bool passed;
    float color_std;
    float contrast_std;
    float avg_color_rgb[3];
    float avg_hue;
    float avg_saturation;
    float avg_value;
    int cloudiness;
    bool is_significant;
    std::string dominant_type;
    std::string error; // To handle "Could not load image"
};

PrefilterResult prefilter_image(const cv::Mat& img, int cloudiness_threshold = 50, int white_threshold = 100, int color_threshold = 30, int contrast_threshold = 20);

nlohmann::ordered_json PrefilterResultToJson(const PrefilterResult& res);
PrefilterResult PrefilterResultFromJson(const nlohmann::json& j);

#endif // PREFILTERING_H
