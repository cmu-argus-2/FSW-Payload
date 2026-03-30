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
    bool is_significant;
    std::string dominant_type;
    
    // Core metrics
    float color_std;
    float contrast_std;
    float avg_color_rgb[3];
    float avg_hue;
    float avg_saturation;
    float avg_value;
    int cloudiness;
    
    PrefilterResult() 
        : passed(true), is_significant(false), dominant_type(""),
          color_std(0.0f), contrast_std(0.0f),
          avg_hue(0.0f), avg_saturation(0.0f), avg_value(0.0f),
          cloudiness(0) {
        avg_color_rgb[0] = 0.0f;
        avg_color_rgb[1] = 0.0f;
        avg_color_rgb[2] = 0.0f;
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
    result.dominant_type = "";

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



int main(int argc, char* argv[]) {
    // Test directory
    std::string home_dir = std::getenv("HOME");
    fs::path test_dir = fs::path(home_dir) / "Desktop" / "prefilter_test";
    
    if (!fs::exists(test_dir)) {
        std::cerr << "Error: Directory " << test_dir << " does not exist" << std::endl;
        return 1;
    }
    
    // Get all image files
    std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"};
    std::vector<fs::path> image_files;
    
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end()) {
                image_files.push_back(entry.path());
            }
        }
    }
    
    if (image_files.empty()) {
        std::cout << "No image files found in " << test_dir << std::endl;
        return 1;
    }
    
    std::sort(image_files.begin(), image_files.end());
    
    std::cout << "\nTesting " << image_files.size() << " images from " << test_dir << "\n" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    int passed_count = 0;
    int rejected_count = 0;
    
    // Default thresholds for testing
    int cloudiness_threshold = 20;
    int white_threshold = 200;
    int color_threshold = 15;
    int contrast_threshold = 20;
    
    for (const auto& img_file : image_files) {
        try {
            cv::Mat img = cv::imread(img_file.string());
            if (img.empty()) {
                std::cout << "\n" << img_file.filename().string() << std::endl;
                std::cout << "  ERROR: Failed to load image" << std::endl;
                rejected_count++;
                continue;
            }
            
            PrefilterResult result = prefilter_image(img, cloudiness_threshold, white_threshold, color_threshold, contrast_threshold);
            
            std::string status = result.passed ? "PASSED ✓" : "REJECTED ✗";
            if (result.passed) {
                passed_count++;
            } else {
                rejected_count++;
            }
            
            std::cout << "\n" << img_file.filename().string() << std::endl;
            std::cout << "  Status: " << status << std::endl;
            std::cout << "  Type: " << result.dominant_type << std::endl;
            std::cout << "  Significant: " << (result.is_significant ? "yes" : "no") << std::endl;
            std::cout << "  Cloudiness: " << result.cloudiness 
                      << " | Brightness: " << result.avg_value 
                      << " | Color Std: " << result.color_std 
                      << " | Contrast Std: " << result.contrast_std << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "\n" << img_file.filename().string() << std::endl;
            std::cout << "  ERROR: " << e.what() << std::endl;
            rejected_count++;
        }
    }
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "SUMMARY: " << passed_count << " passed, " << rejected_count << " rejected" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
}
