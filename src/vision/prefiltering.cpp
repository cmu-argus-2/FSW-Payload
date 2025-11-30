#include "vision/prefiltering.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

PrefilterResult prefilter_image(const cv::Mat& img, int cloudiness_threshold, int white_threshold, int color_threshold, int contrast_threshold) {
    PrefilterResult result;
    result.passed = false; // Default to false
    result.is_significant = false;
    result.dominant_type = "";

    // Get rbg and hsv versions
    Mat img_rgb;
    cvtColor(img, img_rgb, COLOR_BGR2RGB);

    Mat img_hsv;
    cvtColor(img, img_hsv, COLOR_BGR2HSV);

    // std_dev of color channels
    Scalar mean_rgb, stddev_rgb;
    meanStdDev(img_rgb, mean_rgb, stddev_rgb);
    double avg_color_rgb = (stddev_rgb[0] + stddev_rgb[1] + stddev_rgb[2]) / 3.0;

    // std_dev of contrast in grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    Scalar mean_gray, stddev_gray;
    meanStdDev(gray, mean_gray, stddev_gray);
    double contrast_std = stddev_gray[0];

    // avg color, hue, saturation, value
    Scalar mean_hsv, stddev_hsv;
    meanStdDev(img_hsv, mean_hsv, stddev_hsv);
    double avg_hue = mean_hsv[0];
    double avg_saturation = mean_hsv[1];
    double avg_value = mean_hsv[2];

    // Calculate cloudiness
    double white_pixel_count = 0;
    double total_pixel_count = img.rows * img.cols;
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) { 
            Vec3b pixel = img.at<Vec3b>(i, j);
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
        result.dominant_type = "cloudy";
    }
    // else if (is_blue) {
    //     result.passed = false;
    //     result.dominant_type = "blue";
    // }
    else if (is_black) {
        result.passed = false;
        result.dominant_type = "black";
    }
    else if (is_white) {
        result.passed = false;
        result.dominant_type = "white";
    }
    else if (is_green) {
        result.passed = true;
        result.is_significant = true;
        result.dominant_type = "green";
    }
    else {
        result.passed = true;
        result.dominant_type = "single_color";
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

    return result;
}