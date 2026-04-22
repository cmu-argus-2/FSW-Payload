#ifndef PREFILTERING_H
#define PREFILTERING_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

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

PrefilterResult prefilter_image(const cv::Mat& img, int cloudiness_threshold = 1000, int white_threshold = 2000, int color_threshold = 0, int contrast_threshold = 0);

#endif // PREFILTERING_H