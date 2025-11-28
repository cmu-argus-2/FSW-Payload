#include <iostream>
#include "vision/prefiltering.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./prefilter <path_to_image>" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    std::cout << "Processing: " << image_path << std::endl;

    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Could not load image" << std::endl;
        return -1;
    }

    PrefilterResult res = prefilter_image(img);

    if (!res.error.empty()) {
        std::cerr << "Error: " << res.error << std::endl;
        return -1;
    }

    std::cout << "--- Results ---" << std::endl;
    std::cout << "Passed: " << (res.passed ? "Yes" : "No") << std::endl;
    std::cout << "Dominant Type: " << res.dominant_type << std::endl;
    std::cout << "Cloudiness: " << res.cloudiness << "%" << std::endl;
    std::cout << "Contrast Std Dev: " << res.contrast_std << std::endl;
    std::cout << "Color Std Dev: " << res.color_std << std::endl;
    std::cout << "Avg RGB: (" << res.avg_color_rgb[0] << ", " 
              << res.avg_color_rgb[1] << ", " << res.avg_color_rgb[2] << ")" << std::endl;

    return 0;
}