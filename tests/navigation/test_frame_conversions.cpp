#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

static const std::string SAMPLE_JSON_PATH =
    std::string(ROOT_DIR) + "/tests/sample_data/frame_example.json";


static Frame LoadFrameFromJson()
{
    std::ifstream f(SAMPLE_JSON_PATH);
    EXPECT_TRUE(f.is_open()) << "Could not open: " << SAMPLE_JSON_PATH;
    nlohmann::json j;
    f >> j;
    Frame frame;
    frame.fromJson(j);
    return frame;
}

static cv::Mat MakeK(double fx, double fy, double cx, double cy)
{
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = fx;
    K.at<double>(1, 1) = fy;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 2) = cy;
    return K;
}

static cv::Mat ZeroDistortion()
{
    return cv::Mat::zeros(1, 5, CV_64F);
}

// ── PixelToBodyBearing ────────────────────────────────────────────────────────

TEST(PixelToBodyBearingTest, PrincipalPointGivesOpticalAxis)
{
    const double fx = 4000.0, fy = 4000.0, cx = 2304.0, cy = 1296.0;
    cv::Mat K = MakeK(fx, fy, cx, cy);
    cv::Mat D = ZeroDistortion();

    Vector3d bearing = PixelToBodyBearing(static_cast<float>(cx),
                                           static_cast<float>(cy), K, D);

    EXPECT_NEAR(bearing[0], 0.0, 1e-6);
    EXPECT_NEAR(bearing[1], 0.0, 1e-6);
    EXPECT_NEAR(bearing[2], 1.0, 1e-6);
}

// Use the landmark pixel positions from the sample frame JSON and verify
// that PixelToBodyBearing produces a unit vector pointing into the scene.
TEST(PixelToBodyBearingTest, LandmarkPixelsFromJsonAreUnitVectors)
{

    Frame frame = LoadFrameFromJson();
    const double fx = 4000.0, fy = 4000.0, cx = 2304.0, cy = 1296.0;
    cv::Mat K = MakeK(fx, fy, cx, cy);
    cv::Mat D = ZeroDistortion();

    for (const auto& lm : frame.GetLandmarks())
    {
        cv::Vec3d b = PixelToBodyBearing(lm.x, lm.y, K, D);
        EXPECT_NEAR(cv::norm(b), 1.0, 1e-9)
            << "Not a unit vector for landmark at (" << lm.x << ", " << lm.y << ")";
        EXPECT_GT(b[2], 0.0)
            << "Bearing z should be positive (pointing into scene)";
    }
}
