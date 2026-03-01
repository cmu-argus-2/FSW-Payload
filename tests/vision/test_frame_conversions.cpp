#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include "vision/frame.hpp"

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

// ── Frame JSON round-trip ─────────────────────────────────────────────────────

TEST(FrameJsonTest, RoundTrip)
{
    std::ifstream f(SAMPLE_JSON_PATH);
    ASSERT_TRUE(f.is_open()) << "Could not open: " << SAMPLE_JSON_PATH;
    nlohmann::json original;
    f >> original;

    Frame frame;
    frame.fromJson(original);

    EXPECT_EQ(frame.GetTimestamp(),
              original.at("timestamp").get<std::uint64_t>());
    EXPECT_EQ(frame.GetCamID(),
              original.at("cam_id").get<int>());
    EXPECT_EQ(static_cast<int>(frame.GetProcessingStage()),
              original.at("processing_stage").get<int>());
    EXPECT_EQ(static_cast<int>(frame.GetImageState()),
              original.at("annotation_state").get<int>());
    EXPECT_EQ(frame.GetLandmarks().size(),
              static_cast<size_t>(original.at("detected_landmarks_count").get<int>()));
    EXPECT_EQ(frame.GetRegions().size(),
              static_cast<size_t>(original.at("detected_regions_count").get<int>()));

    nlohmann::json roundtripped = frame.toJson();
    EXPECT_EQ(roundtripped.at("timestamp"),         original.at("timestamp"));
    EXPECT_EQ(roundtripped.at("cam_id"),            original.at("cam_id"));
    EXPECT_EQ(roundtripped.at("processing_stage"),  original.at("processing_stage"));
    EXPECT_EQ(roundtripped.at("annotation_state"),  original.at("annotation_state"));
}

// ── PixelToBodyBearing ────────────────────────────────────────────────────────

TEST(PixelToBodyBearingTest, PrincipalPointGivesOpticalAxis)
{
    const double fx = 4000.0, fy = 4000.0, cx = 2304.0, cy = 1296.0;
    cv::Mat K = MakeK(fx, fy, cx, cy);
    cv::Mat D = ZeroDistortion();

    cv::Vec3d bearing = PixelToBodyBearing(static_cast<float>(cx),
                                           static_cast<float>(cy), K, D);

    EXPECT_NEAR(bearing[0], 0.0, 1e-6);
    EXPECT_NEAR(bearing[1], 0.0, 1e-6);
    EXPECT_NEAR(bearing[2], 1.0, 1e-6);
}

TEST(PixelToBodyBearingTest, OffAxisPixelCorrectDirection)
{
    // One focal length right of principal point:
    //   normalised coords → (1, 0), bearing (1,0,1)/‖·‖ = (1/√2, 0, 1/√2)
    const double fx = 4000.0, fy = 4000.0, cx = 2304.0, cy = 1296.0;
    cv::Mat K = MakeK(fx, fy, cx, cy);
    cv::Mat D = ZeroDistortion();

    const double expected = 1.0 / std::sqrt(2.0);
    cv::Vec3d bearing = PixelToBodyBearing(static_cast<float>(cx + fx),
                                           static_cast<float>(cy), K, D);

    EXPECT_NEAR(bearing[0], expected, 1e-6);
    EXPECT_NEAR(bearing[1], 0.0,     1e-6);
    EXPECT_NEAR(bearing[2], expected, 1e-6);
}

TEST(PixelToBodyBearingTest, LandmarkPixelsFromJsonAreUnitVectors)
{
    // Use the landmark pixel positions from the sample frame JSON and verify
    // that PixelToBodyBearing produces a unit vector pointing into the scene.
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

// ── LatLonToECI ───────────────────────────────────────────────────────────────

// Timestamp from the JSON (microseconds) converted to seconds for all ECI tests.
static double JsonTimestampSeconds()
{
    return static_cast<double>(LoadFrameFromJson().GetTimestamp()) / 1.0e6;
}

TEST(LatLonToECITest, EquatorialMagnitudeEqualsEarthRadius)
{
    const double t = JsonTimestampSeconds();
    cv::Vec3d r = LatLonToECI(t, 0.0, 0.0, 0.0);
    EXPECT_NEAR(cv::norm(r), 6378137.0, 1.0);
}

TEST(LatLonToECITest, PolarZComponentEqualsPolarRadius)
{
    // ECEF→ECI only rotates around Z, so z_eci == z_ecef == b at the pole.
    const double t = JsonTimestampSeconds();
    const double b = 6378137.0 * std::sqrt(1.0 - 0.00669437999014);
    cv::Vec3d r = LatLonToECI(t, 90.0, 0.0, 0.0);
    EXPECT_NEAR(r[2], b,   1.0);
    EXPECT_NEAR(r[0], 0.0, 1.0);
    EXPECT_NEAR(r[1], 0.0, 1.0);
}

TEST(LatLonToECITest, AltitudeIncreasesRadius)
{
    const double t     = JsonTimestampSeconds();
    const double alt_m = 500000.0;
    cv::Vec3d r_surface = LatLonToECI(t, 45.0, 90.0, 0.0);
    cv::Vec3d r_orbit   = LatLonToECI(t, 45.0, 90.0, alt_m);
    EXPECT_GT(cv::norm(r_orbit), cv::norm(r_surface));
    EXPECT_NEAR(cv::norm(r_orbit) - cv::norm(r_surface), alt_m, 1000.0);
}

TEST(LatLonToECITest, MagnitudePreservedAcrossTimestamps)
{
    const double t = JsonTimestampSeconds();
    cv::Vec3d r1 = LatLonToECI(t,          35.0, 139.0, 400000.0);
    cv::Vec3d r2 = LatLonToECI(t + 3600.0, 35.0, 139.0, 400000.0);
    EXPECT_NEAR(cv::norm(r1), cv::norm(r2), 1.0);
}

TEST(LatLonToECITest, EarthRotationChangesXYNotZ)
{
    // Earth rotates ~15°/hour: ECI x,y change but z is unaffected at the equator.
    const double t = JsonTimestampSeconds();
    cv::Vec3d r1 = LatLonToECI(t,          0.0, 0.0, 0.0);
    cv::Vec3d r2 = LatLonToECI(t + 3600.0, 0.0, 0.0, 0.0);
    EXPECT_NE(r1[0], r2[0]);
    EXPECT_NE(r1[1], r2[1]);
    EXPECT_NEAR(r1[2], r2[2], 1e-6);
}
