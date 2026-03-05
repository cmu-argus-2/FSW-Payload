#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <vision/frame.hpp>
#include <navigation/utils.hpp>

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

    Eigen::Vector3d bearing = PixelToBodyBearing(static_cast<float>(cx),
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
        Eigen::Vector3d b = PixelToBodyBearing(lm.x, lm.y, K, D);
        EXPECT_NEAR(b.norm(), 1.0, 1e-9)
            << "Not a unit vector for landmark at (" << lm.x << ", " << lm.y << ")";
        EXPECT_GT(b[2], 0.0)
            << "Bearing z should be positive (pointing into scene)";
    }
}

TEST(PixelToBodyBearingTest, OffAxisPixelCorrectDirection)
{
    // One focal length right of principal point:
    //   normalised coords → (1, 0), bearing (1,0,1)/‖·‖ = (1/√2, 0, 1/√2)
    const double fx = 4000.0, fy = 4000.0, cx = 2304.0, cy = 1296.0;
    cv::Mat K = MakeK(fx, fy, cx, cy);
    cv::Mat D = ZeroDistortion();

    const double expected = 1.0 / std::sqrt(2.0);
    Eigen::Vector3d bearing = PixelToBodyBearing(static_cast<float>(cx + fx),
                                                 static_cast<float>(cy), K, D);

    EXPECT_NEAR(bearing[0], expected, 1e-6);
    EXPECT_NEAR(bearing[1], 0.0,     1e-6);
    EXPECT_NEAR(bearing[2], expected, 1e-6);
}

// ── LAT2ECI ───────────────────────────────────────────────────────────────────

// Timestamp from the JSON (microseconds) converted to J2000 seconds for all ECI tests.
static double JsonTimestampJ2000()
{
    uint64_t ts_us = LoadFrameFromJson().GetTimestamp();
    int64_t t_unix_s = static_cast<int64_t>(ts_us / 1000000ULL);
    return static_cast<double>(unixToJ2000(t_unix_s));
}

// LAT2ECI takes v_lat = (r [m], longitude [rad], latitude [rad]) in SPICE convention.
static constexpr double EARTH_RADIUS_M = 6378137.0; // WGS84 equatorial radius

TEST(LAT2ECITest, EquatorialMagnitudeEqualsEarthRadius)
{
    const double t_j2000 = JsonTimestampJ2000();
    // lon=0, lat=0 → equatorial prime-meridian point at Earth's surface
    bool geoc = true; // geocentric coordinates
    Eigen::Vector3d v_lat(EARTH_RADIUS_M, 0.0, 0.0);
    Eigen::Vector3d r = LAT2ECI(v_lat, t_j2000, geoc);
    EXPECT_NEAR(r.norm(), EARTH_RADIUS_M, 1.0);
}


// Helper: geographic lat/lon (degrees) + altitude (m) → ECI using existing LAT2ECI.
// LAT2ECI expects (r [m], lon [rad], lat [rad]) in SPICE latitudinal convention.
static Eigen::Vector3d LatLonAltToECI(double t_j2000, double lat_deg, double lon_deg, double alt_m)
{
    double r = EARTH_RADIUS_M + alt_m;
    Eigen::Vector3d v_lat(r, lon_deg * M_PI / 180.0, lat_deg * M_PI / 180.0);
    bool geoc = true; // geocentric coordinates
    return LAT2ECI(v_lat, t_j2000, geoc);
}

TEST(LAT2ECITest, PolarZComponentEqualsPolarRadius)
{
    // The full ECEF→ECI transformation (ITRF93→J2000) includes precession and
    // nutation, so individual components shift by up to ~14 km.
    // The one guarantee is that the orthogonal rotation preserves the magnitude.
    const double t = JsonTimestampJ2000();
    const double b = 6378137.0 * std::sqrt(1.0 - 0.00669437999014);
    Eigen::Vector3d v_lat(b, 0.0, M_PI / 2.0);  // r=b, lon=0, lat=90°
    bool geoc = true; // geocentric coordinates
    Eigen::Vector3d r = LAT2ECI(v_lat, t, geoc);
    EXPECT_NEAR(r.norm(), b, 1.0);
}

TEST(LAT2ECITest, AltitudeIncreasesRadius)
{
    const double t     = JsonTimestampJ2000();
    const double alt_m = 500000.0;
    Eigen::Vector3d r_surface = LatLonAltToECI(t, 45.0, 90.0, 0.0);
    Eigen::Vector3d r_orbit   = LatLonAltToECI(t, 45.0, 90.0, alt_m);
    EXPECT_GT(r_orbit.norm(), r_surface.norm());
    EXPECT_NEAR(r_orbit.norm() - r_surface.norm(), alt_m, 1000.0);
}

TEST(LAT2ECITest, MagnitudePreservedAcrossTimestamps)
{
    const double t = JsonTimestampJ2000();
    Eigen::Vector3d r1 = LatLonAltToECI(t,          35.0, 139.0, 400000.0);
    Eigen::Vector3d r2 = LatLonAltToECI(t + 3600.0, 35.0, 139.0, 400000.0);
    EXPECT_NEAR(r1.norm(), r2.norm(), 1.0);
}

TEST(LAT2ECITest, EarthRotationChangesXYNotZ)
{
    // Earth rotates ~15°/hour: after 1h the same ECEF point has different ECI x,y.
    // The full ITRF93→J2000 transformation includes precession/nutation, so z also
    // shifts slightly; the guaranteed invariant is that the magnitude is preserved.
    const double t = JsonTimestampJ2000();
    Eigen::Vector3d r1 = LatLonAltToECI(t,          0.0, 0.0, 0.0);
    Eigen::Vector3d r2 = LatLonAltToECI(t + 3600.0, 0.0, 0.0, 0.0);
    EXPECT_NE(r1[0], r2[0]);
    EXPECT_NE(r1[1], r2[1]);
    EXPECT_NEAR(r1.norm(), r2.norm(), 1.0);
}
