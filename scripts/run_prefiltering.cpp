#include <iostream>
#include "vision/prefiltering.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./RUN_PREFILTERING <image_path> <output_folder>" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string output_folder = argv[2];

    std::cout << "Processing: " << image_path << std::endl;

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Could not load image" << std::endl;
        return 1;
    }

    PrefilterResult res = prefilter_image(img, 70, 220, 40, 40);

    if (res.error[0] != '\0') {
        std::cerr << "Error: " << res.error << std::endl;
        return 1;
    }

    std::cout << "Passed: "       << (res.passed ? "Yes" : "No")         << std::endl;
    std::cout << "Significant: "  << (res.is_significant ? "Yes" : "No") << std::endl;
    std::cout << "Dominant Type: "<< res.dominant_type                    << std::endl;
    std::cout << "Hue: "          << res.avg_hue                          << std::endl;
    std::cout << "Saturation: "   << res.avg_saturation                   << std::endl;
    std::cout << "Brightness: "   << res.avg_value                        << std::endl;
    std::cout << "Color Std: "    << res.color_std                        << std::endl;
    std::cout << "Contrast Std: " << res.contrast_std                     << std::endl;
    std::cout << "Cloudiness: "   << res.cloudiness                       << std::endl;
    std::cout << "Avg RGB: ("     << res.avg_color_rgb[0] << ", "
                                  << res.avg_color_rgb[1] << ", "
                                  << res.avg_color_rgb[2] << ")"          << std::endl;

    return 0;
}