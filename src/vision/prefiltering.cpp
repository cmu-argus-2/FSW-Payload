#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

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
    char dominant_type[32];
    char error[128];

    PrefilterResult() 
        : passed(false), is_significant(false),
          color_std(0.0f), contrast_std(0.0f),
          avg_hue(0.0f), avg_saturation(0.0f), avg_value(0.0f),
          cloudiness(0) {
        avg_color_rgb[0] = 0.0f;
        avg_color_rgb[1] = 0.0f;
        avg_color_rgb[2] = 0.0f;
        dominant_type[0] = '\0';
        error[0] = '\0';
    }
};

PrefilterResult prefilter_image(
    const cv::Mat& img,
    int cloudiness_threshold,
    int white_threshold,
    int color_threshold,
    int contrast_threshold
) {
    PrefilterResult result;
    result.passed = false; // Default to false
    result.is_significant = false;
    result.dominant_type[0] = '\0';

    // Get RGB and HSV versions
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat img_hsv;
    cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);

    // std_dev of color channels
    cv::Scalar mean_rgb, stddev_rgb;
    cv::meanStdDev(img_rgb, mean_rgb, stddev_rgb);
    double avg_color_rgb = (stddev_rgb[0] + stddev_rgb[1] + stddev_rgb[2]) / 3.0;

    // std_dev of contrast in grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    cv::Scalar mean_gray, stddev_gray;
    cv::meanStdDev(gray, mean_gray, stddev_gray);
    double contrast_std = stddev_gray[0];

    // avg color, hue, saturation, value
    cv::Scalar mean_hsv, stddev_hsv;
    cv::meanStdDev(img_hsv, mean_hsv, stddev_hsv);
    double avg_hue = mean_hsv[0];
    double avg_saturation = mean_hsv[1];
    double avg_value = mean_hsv[2];

    // Calculate cloudiness
    double white_pixel_count = 0;
    double total_pixel_count = img.rows * img.cols;
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) { 
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            if (pixel[0] > white_threshold && pixel[1] > white_threshold && pixel[2] > white_threshold) {
                white_pixel_count++;
            }
        }
    }
    int cloudiness = static_cast<int>((white_pixel_count / total_pixel_count) * 100);

    // Basic filtering logic
    bool has_variety = (avg_color_rgb > color_threshold) && (contrast_std > contrast_threshold);
    bool is_blue = (90 < avg_hue && avg_hue < 130) && (avg_saturation > 50);
    bool is_black = (avg_value < 50);
    bool is_white = (avg_value > 200) && (avg_saturation < 30);
    bool is_green = (34 < avg_hue && avg_hue < 85) && (avg_saturation > 50);
    bool is_cloudy = cloudiness > cloudiness_threshold;

    if (has_variety) {
        result.passed = true;
    }
    else if (is_cloudy) {
        result.passed = false;
        strncpy(result.dominant_type, "cloudy", sizeof(result.dominant_type));
    }
    // else if (is_blue) {
    //     result.passed = false;
    //     result.dominant_type = "blue";
    // }
    else if (is_black) {
        result.passed = false;
        strncpy(result.dominant_type, "black", sizeof(result.dominant_type));
    }
    else if (is_white) {
        result.passed = false;
        strncpy(result.dominant_type, "white", sizeof(result.dominant_type));
    }
    else if (is_green) {
        result.passed = true;
        result.is_significant = true;
        strncpy(result.dominant_type, "green", sizeof(result.dominant_type));
    }
    else {
        result.passed = false;
        strncpy(result.dominant_type, "single_color", sizeof(result.dominant_type));
    }

    result.color_std = static_cast<float>(avg_color_rgb);
    result.contrast_std = static_cast<float>(contrast_std);
    result.avg_color_rgb[0] = mean_rgb[0];
    result.avg_color_rgb[1] = mean_rgb[1];
    result.avg_color_rgb[2] = mean_rgb[2];
    result.avg_hue = static_cast<float>(avg_hue);
    result.avg_saturation = static_cast<float>(avg_saturation);
    result.avg_value = static_cast<float>(avg_value);
    result.cloudiness = cloudiness;

    // std::cout << "  Type:        " << (result.dominant_type.empty() ? "none" : result.dominant_type) << std::endl;
    // std::cout << "  Passed:      " << (result.passed ? "yes" : "no") << std::endl;
    // std::cout << "  Significant: " << (result.is_significant ? "yes" : "no") << std::endl;
    // std::cout << "  --- Color ---" << std::endl;
    // std::cout << "  Hue:         " << result.avg_hue << " (blue=90-130, green=34-85)" << std::endl;
    // std::cout << "  Saturation:  " << result.avg_saturation << " (threshold >50 for color detection)" << std::endl;
    // std::cout << "  Brightness:  " << result.avg_value << " (black<50, white>200)" << std::endl;
    // std::cout << "  Color Std:   " << result.color_std << " (threshold=" << color_threshold << ")" << std::endl;
    // std::cout << "  Avg RGB:     (" << result.avg_color_rgb[0] << ", " << result.avg_color_rgb[1] << ", " << result.avg_color_rgb[2] << ")" << std::endl;
    // std::cout << "  --- Texture ---" << std::endl;
    // std::cout << "  Contrast Std:" << result.contrast_std << " (threshold=" << contrast_threshold << ")" << std::endl;
    // std::cout << "  Cloudiness:  " << result.cloudiness << "% (threshold=" << cloudiness_threshold << "%, white_px>" << white_threshold << ")" << std::endl;
    // std::cout << "  --- Flags ---" << std::endl;
    // std::cout << "  has_variety: " << (has_variety ? "yes" : "no") << " (color>" << color_threshold << " && contrast>" << contrast_threshold << ")" << std::endl;
    // std::cout << "  is_green:    " << (is_green ? "yes" : "no") << std::endl;
    // std::cout << "  is_cloudy:   " << (is_cloudy ? "yes" : "no") << std::endl;
    // std::cout << "  is_black:    " << (is_black ? "yes" : "no") << std::endl;
    // std::cout << "  is_white:    " << (is_white ? "yes" : "no") << std::endl;

    return result;
}

nlohmann::ordered_json PrefilterResultToJson(const PrefilterResult& res)
{
    return nlohmann::ordered_json{
        {"passed",         res.passed},
        {"cloudiness",     res.cloudiness},
        {"color_std",      res.color_std},
        {"contrast_std",   res.contrast_std},
        {"avg_color_rgb",  nlohmann::ordered_json::array({res.avg_color_rgb[0], res.avg_color_rgb[1], res.avg_color_rgb[2]})},
        {"avg_hue",        res.avg_hue},
        {"avg_saturation", res.avg_saturation},
        {"avg_value",      res.avg_value},
        {"is_significant", res.is_significant},
        {"dominant_type",  res.dominant_type},
        {"error",          res.error}
    };
}

PrefilterResult PrefilterResultFromJson(const nlohmann::json& j)
{
    PrefilterResult res{};
    res.passed         = j.at("passed").get<bool>();
    res.cloudiness     = j.at("cloudiness").get<int>();
    res.color_std      = j.at("color_std").get<float>();
    res.contrast_std   = j.at("contrast_std").get<float>();
    const auto& rgb    = j.at("avg_color_rgb");
    res.avg_color_rgb[0] = rgb[0].get<float>();
    res.avg_color_rgb[1] = rgb[1].get<float>();
    res.avg_color_rgb[2] = rgb[2].get<float>();
    res.avg_hue        = j.at("avg_hue").get<float>();
    res.avg_saturation = j.at("avg_saturation").get<float>();
    res.avg_value      = j.at("avg_value").get<float>();
    res.is_significant = j.at("is_significant").get<bool>();
    res.dominant_type  = j.at("dominant_type").get<std::string>();
    res.error          = j.at("error").get<std::string>();
    return res;
}