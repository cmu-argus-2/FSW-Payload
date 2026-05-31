#include "vision/prefiltering.hpp"
#include <opencv2/imgproc.hpp>

PrefilterResult prefilter_image(const cv::Mat& img,
                                int cloudiness_threshold,
                                int white_threshold,
                                int color_threshold,
                                int contrast_threshold)
{
    PrefilterResult result{};
    result.passed = false;
    result.is_significant = false;
    result.dominant_type.clear();
    result.error.clear();

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat img_hsv;
    cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV);

    cv::Scalar mean_rgb, stddev_rgb;
    cv::meanStdDev(img_rgb, mean_rgb, stddev_rgb);
    const double avg_color_std = (stddev_rgb[0] + stddev_rgb[1] + stddev_rgb[2]) / 3.0;

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Scalar mean_gray, stddev_gray;
    cv::meanStdDev(gray, mean_gray, stddev_gray);
    const double contrast_std = stddev_gray[0];

    cv::Scalar mean_hsv, stddev_hsv;
    cv::meanStdDev(img_hsv, mean_hsv, stddev_hsv);
    const double avg_hue = mean_hsv[0];
    const double avg_saturation = mean_hsv[1];
    const double avg_value = mean_hsv[2];

    double white_pixel_count = 0;
    const double total_pixel_count = static_cast<double>(img.rows) * static_cast<double>(img.cols);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            const cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            if (pixel[0] > white_threshold && pixel[1] > white_threshold && pixel[2] > white_threshold) {
                white_pixel_count++;
            }
        }
    }
    const int cloudiness = static_cast<int>((white_pixel_count / total_pixel_count) * 100.0);

    const bool has_variety = (avg_color_std > color_threshold) && (contrast_std > contrast_threshold);
    const bool is_black = (avg_value < 50.0);
    const bool is_white = (avg_value > 200.0) && (avg_saturation < 30.0);
    const bool is_green = (34.0 < avg_hue && avg_hue < 85.0) && (avg_saturation > 50.0);
    const bool is_cloudy = cloudiness > cloudiness_threshold;

    if (has_variety) {
        result.passed = true;
    } else if (is_cloudy) {
        result.passed = false;
        result.dominant_type = "cloudy";
    } else if (is_black) {
        result.passed = false;
        result.dominant_type = "black";
    } else if (is_white) {
        result.passed = false;
        result.dominant_type = "white";
    } else if (is_green) {
        result.passed = true;
        result.is_significant = true;
        result.dominant_type = "green";
    } else {
        result.passed = true;
        result.dominant_type = "single_color";
    }

    result.color_std = static_cast<float>(avg_color_std);
    result.contrast_std = static_cast<float>(contrast_std);
    result.avg_color_rgb[0] = static_cast<float>(mean_rgb[0]);
    result.avg_color_rgb[1] = static_cast<float>(mean_rgb[1]);
    result.avg_color_rgb[2] = static_cast<float>(mean_rgb[2]);
    result.avg_hue = static_cast<float>(avg_hue);
    result.avg_saturation = static_cast<float>(avg_saturation);
    result.avg_value = static_cast<float>(avg_value);
    result.cloudiness = cloudiness;

    return result;
}

nlohmann::ordered_json PrefilterResultToJson(const PrefilterResult& res)
{
    return nlohmann::ordered_json{
        {"passed", res.passed},
        {"cloudiness", res.cloudiness},
        {"color_std", res.color_std},
        {"contrast_std", res.contrast_std},
        {"avg_color_rgb", nlohmann::ordered_json::array({res.avg_color_rgb[0], res.avg_color_rgb[1], res.avg_color_rgb[2]})},
        {"avg_hue", res.avg_hue},
        {"avg_saturation", res.avg_saturation},
        {"avg_value", res.avg_value},
        {"is_significant", res.is_significant},
        {"dominant_type", res.dominant_type},
        {"error", res.error}
    };
}

PrefilterResult PrefilterResultFromJson(const nlohmann::json& j)
{
    PrefilterResult res{};
    res.passed = j.at("passed").get<bool>();
    res.cloudiness = j.at("cloudiness").get<int>();
    res.color_std = j.at("color_std").get<float>();
    res.contrast_std = j.at("contrast_std").get<float>();
    const auto& rgb = j.at("avg_color_rgb");
    res.avg_color_rgb[0] = rgb[0].get<float>();
    res.avg_color_rgb[1] = rgb[1].get<float>();
    res.avg_color_rgb[2] = rgb[2].get<float>();
    res.avg_hue = j.at("avg_hue").get<float>();
    res.avg_saturation = j.at("avg_saturation").get<float>();
    res.avg_value = j.at("avg_value").get<float>();
    res.is_significant = j.at("is_significant").get<bool>();
    res.dominant_type = j.at("dominant_type").get<std::string>();
    res.error = j.at("error").get<std::string>();
    return res;
}
