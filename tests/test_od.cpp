#include <gtest/gtest.h>

#include "configuration.hpp"
#include "navigation/od.hpp"
#include "navigation/pose_dynamics.hpp"
#include "navigation/utils.hpp"
#include "vision/regions.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <string>

namespace fs = std::filesystem;

namespace
{
constexpr int kLDNetedStage = static_cast<int>(ProcessingStage::LDNeted);
constexpr int kRegion17R = static_cast<int>(RegionID::R_17R);
constexpr uint64_t kBaseUnixMs =
    static_cast<uint64_t>(J2000_EPOCH_UNIX_S + 1000) * 1000ULL;

struct ODTest : ::testing::Test
{
    fs::path root;

    void SetUp() override
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root = fs::temp_directory_path() / ("fsw_od_test_" + std::to_string(stamp));
        fs::create_directories(root);
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    fs::path makeDir(const std::string& name)
    {
        fs::path dir = root / name;
        fs::create_directories(dir);
        return dir;
    }
};

void WriteText(const fs::path& path, const std::string& text)
{
    std::ofstream f(path);
    ASSERT_TRUE(f.is_open()) << path;
    f << text;
}

void WriteFrameJson(const fs::path& dir,
                    int index,
                    int stage,
                    int landmark_count,
                    int ld_version = 2,
                    int region_id = kRegion17R,
                    int class_id = 0,
                    bool include_inference = true,
                    bool include_ld_version = true)
{
    const uint64_t timestamp_ms = kBaseUnixMs + static_cast<uint64_t>(index) * 1000ULL;
    std::ofstream f(dir / ("frame_" + std::to_string(index) + ".json"));
    ASSERT_TRUE(f.is_open());
    f << "{\n"
      << "  \"processing_stage\": " << stage << ",\n"
      << "  \"detected_landmarks_count\": " << landmark_count << ",\n"
      << "  \"timestamp\": " << timestamp_ms << ",\n"
      << "  \"cam_id\": 0";

    if (include_inference) {
        f << ",\n  \"inference_results\": {\n";
        if (include_ld_version) {
            f << "    \"ldnet_version\": " << ld_version << ",\n";
        }
        f << "    \"detected_landmarks_count\": " << landmark_count << ",\n"
          << "    \"landmarks\": [\n";
        for (int i = 0; i < landmark_count; ++i) {
            if (i > 0) f << ",\n";
            f << "      {\"landmark_" << i << "\": {"
              << "\"x\": 1000.0, "
              << "\"y\": 1000.0, "
              << "\"height\": 30.0, "
              << "\"width\": 30.0, "
              << "\"confidence\": 0.9, "
              << "\"class_id\": " << class_id << ", "
              << "\"region_id\": " << region_id
              << "}}";
        }
        f << "\n    ]\n  }";
    }

    f << "\n}\n";
}

void WriteMalformedFrameJson(const fs::path& dir)
{
    WriteText(dir / "frame_0.json", "{ definitely not json\n");
}

void WriteImuCsv(const fs::path& dir,
                 const std::vector<std::array<double, 4>>& rows,
                 bool malformed = false)
{
    std::ofstream f(dir / "imu_data.csv");
    ASSERT_TRUE(f.is_open());
    f << std::fixed << std::setprecision(9);
    f << "timestamp_ms,gyro_x_dps,gyro_y_dps,gyro_z_dps\n";
    if (malformed) {
        f << "bad,row\n";
        return;
    }
    for (const auto& row : rows) {
        f << row[0] << ',' << row[1] << ',' << row[2] << ',' << row[3] << '\n';
    }
}

void WriteLandmarkCsv(const fs::path& dir,
                      const std::vector<std::array<double, 9>>& rows,
                      bool malformed = false)
{
    std::ofstream f(dir / "landmark_measurements.csv");
    ASSERT_TRUE(f.is_open());
    f << std::fixed << std::setprecision(9);
    f << "timestamp_ms,bearing_x,bearing_y,bearing_z,eci_x_km,eci_y_km,eci_z_km,group,sigma\n";
    if (malformed) {
        f << "not,enough,columns\n";
        return;
    }
    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) f << ',';
            f << row[i];
        }
        f << '\n';
    }
}

CameraCalibration MakeCalibration()
{
    CameraCalibration calibration;
    calibration.camera_matrix = (cv::Mat_<double>(3, 3) <<
        1000.0, 0.0, 1000.0,
        0.0, 1000.0, 1000.0,
        0.0, 0.0, 1.0);
    calibration.dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    for (auto& rot : calibration.cam_to_body) {
        rot = cv::Mat::eye(3, 3, CV_64F);
    }
    return calibration;
}

std::vector<std::array<double, 4>> ValidImuRows()
{
    return {{
        {static_cast<double>(kBaseUnixMs - 1000), 0.0, 0.0, 0.0},
        {static_cast<double>(kBaseUnixMs), 0.0, 0.0, 0.0},
        {static_cast<double>(kBaseUnixMs + 1000), 0.0, 0.0, 0.0},
        {static_cast<double>(kBaseUnixMs + 2000), 0.0, 0.0, 0.0},
        {static_cast<double>(kBaseUnixMs + 3000), 0.0, 0.0, 0.0},
    }};
}

std::array<double, 9> LandmarkRow(double timestamp_ms, int group)
{
    return {timestamp_ms, 1.0, 0.0, 0.0, 7000.0, 0.0, 0.0,
            static_cast<double>(group), 0.01};
}
} // namespace

TEST_F(ODTest, ReadODConfigValidConfigReturnsOK)
{
    const ODConfigResult result = ReadODConfig("config/od.toml");
    EXPECT_EQ(result.code, ErrorCode::OK);
    EXPECT_GT(result.config.batch_opt.max_iterations, 0u);
}

TEST_F(ODTest, ReadODConfigNonexistentPathReturnsFileDoesNotExist)
{
    const ODConfigResult result = ReadODConfig((root / "missing.toml").string());
    EXPECT_EQ(result.code, ErrorCode::FILE_DOES_NOT_EXIST);
}

TEST_F(ODTest, ReadODConfigMalformedTomlReturnsFileNotAvailable)
{
    const fs::path path = root / "bad.toml";
    WriteText(path, "[INIT\n");
    const ODConfigResult result = ReadODConfig(path.string());
    EXPECT_EQ(result.code, ErrorCode::FILE_NOT_AVAILABLE);
}

TEST_F(ODTest, ReadODConfigMissingRequiredSectionsReturnsFileNotAvailable)
{
    const fs::path path = root / "missing_sections.toml";
    WriteText(path, "[INIT]\ncollection_period = 1\n");
    const ODConfigResult result = ReadODConfig(path.string());
    EXPECT_EQ(result.code, ErrorCode::FILE_NOT_AVAILABLE);
}

TEST_F(ODTest, InspectDatasetForODMissingFolderReturnsDatasetNotAvailable)
{
    EXPECT_EQ(InspectDatasetForOD((root / "missing").string()),
              ODStage::DATASET_NOT_AVAILABLE);
}

TEST_F(ODTest, InspectDatasetForODEmptyFolderReturnsDatasetNotProcessed)
{
    EXPECT_EQ(InspectDatasetForOD(makeDir("empty").string()),
              ODStage::DATASET_NOT_PROCESSED);
}

TEST_F(ODTest, InspectDatasetForODMalformedFrameReturnsDatasetNotProcessed)
{
    const fs::path dir = makeDir("malformed_frame");
    WriteMalformedFrameJson(dir);
    EXPECT_EQ(InspectDatasetForOD(dir.string()), ODStage::DATASET_NOT_PROCESSED);
}

TEST_F(ODTest, InspectDatasetForODFrameNotLDNetedReturnsDatasetNotProcessed)
{
    const fs::path dir = makeDir("not_ldneted");
    WriteFrameJson(dir, 0, static_cast<int>(ProcessingStage::RCNeted), 1);
    EXPECT_EQ(InspectDatasetForOD(dir.string()), ODStage::DATASET_NOT_PROCESSED);
}

TEST_F(ODTest, InspectDatasetForODLDNetedNoLandmarksReturnsDatasetNotProcessed)
{
    const fs::path dir = makeDir("no_landmarks");
    WriteFrameJson(dir, 0, kLDNetedStage, 0);
    EXPECT_EQ(InspectDatasetForOD(dir.string()), ODStage::DATASET_NOT_PROCESSED);
}

TEST_F(ODTest, InspectDatasetForODLDNetedWithLandmarksReturnsDatasetProcessed)
{
    const fs::path dir = makeDir("processed");
    WriteFrameJson(dir, 0, kLDNetedStage, 1);
    EXPECT_EQ(InspectDatasetForOD(dir.string()), ODStage::DATASET_PROCESSED);
}

TEST_F(ODTest, InspectDatasetForODLandmarkMeasurementsCsvExistsReturnsMeasurementsReady)
{
    const fs::path dir = makeDir("measurements_ready");
    WriteLandmarkCsv(dir, {});
    EXPECT_EQ(InspectDatasetForOD(dir.string()), ODStage::MEASUREMENTS_READY);
}

TEST_F(ODTest, IsODPossibleMissingFolderReturnsFalse)
{
    OD od;
    EXPECT_FALSE(od.IsODPossible((root / "missing").string()));
}

TEST_F(ODTest, IsODPossibleMissingImuCsvReturnsFalse)
{
    OD od;
    const fs::path dir = makeDir("missing_imu");
    WriteFrameJson(dir, 0, kLDNetedStage, 1);
    WriteFrameJson(dir, 1, kLDNetedStage, 1);
    EXPECT_FALSE(od.IsODPossible(dir.string()));
}

TEST_F(ODTest, IsODPossibleNoFrameJsonReturnsFalse)
{
    OD od;
    const fs::path dir = makeDir("no_frames");
    WriteImuCsv(dir, ValidImuRows());
    EXPECT_FALSE(od.IsODPossible(dir.string()));
}

TEST_F(ODTest, IsODPossibleOnlyOneLDNetedLandmarkFrameReturnsFalse)
{
    OD od;
    const fs::path dir = makeDir("one_frame");
    WriteFrameJson(dir, 0, kLDNetedStage, 1);
    WriteImuCsv(dir, ValidImuRows());
    EXPECT_FALSE(od.IsODPossible(dir.string()));
}

TEST_F(ODTest, IsODPossibleTwoFramesNoImuSpanReturnsFalse)
{
    OD od;
    const fs::path dir = makeDir("bad_span");
    WriteFrameJson(dir, 0, kLDNetedStage, 1);
    WriteFrameJson(dir, 1, kLDNetedStage, 1);
    WriteImuCsv(dir, {{{static_cast<double>(kBaseUnixMs - 10000), 0.0, 0.0, 0.0}}});
    EXPECT_FALSE(od.IsODPossible(dir.string()));
}

TEST_F(ODTest, IsODPossibleTwoFramesWithImuSpanReturnsTrue)
{
    OD od;
    const fs::path dir = makeDir("valid_possible");
    WriteFrameJson(dir, 0, kLDNetedStage, 1);
    WriteFrameJson(dir, 1, kLDNetedStage, 1);
    WriteImuCsv(dir, ValidImuRows());
    EXPECT_TRUE(od.IsODPossible(dir.string()));
}

TEST_F(ODTest, DatasetPrepareMissingFolderReturnsFileDoesNotExist)
{
    OD od;
    EXPECT_EQ(od.DatasetPrepare((root / "missing").string(), MakeCalibration()),
              ErrorCode::FILE_DOES_NOT_EXIST);
}

TEST_F(ODTest, DatasetPrepareNoUsableLDVersionReturnsODMeasNotValid)
{
    OD od;
    const fs::path dir = makeDir("no_usable_ld_version");
    WriteFrameJson(dir, 0, static_cast<int>(ProcessingStage::RCNeted), 1);
    EXPECT_EQ(od.DatasetPrepare(dir.string(), MakeCalibration()),
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, DatasetPrepareUsableFrameMissingLDVersionReturnsODMeasNotValid)
{
    OD od;
    const fs::path dir = makeDir("missing_ld_version");
    WriteFrameJson(dir, 0, kLDNetedStage, 1, 2, kRegion17R, 0, true, false);
    EXPECT_EQ(od.DatasetPrepare(dir.string(), MakeCalibration()),
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, DatasetPrepareInvalidLDVersionReturnsODMeasNotValid)
{
    OD od;
    const fs::path dir = makeDir("invalid_ld_version");
    WriteFrameJson(dir, 0, kLDNetedStage, 1, 0);
    EXPECT_EQ(od.DatasetPrepare(dir.string(), MakeCalibration()),
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, DatasetPrepareMixedLDVersionsReturnsODMeasNotValid)
{
    OD od;
    const fs::path dir = makeDir("mixed_ld_versions");
    WriteFrameJson(dir, 0, kLDNetedStage, 1, 2);
    WriteFrameJson(dir, 1, kLDNetedStage, 1, 3);
    EXPECT_EQ(od.DatasetPrepare(dir.string(), MakeCalibration()),
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, DatasetPrepareMissingBoundingBoxesCsvReturnsODMeasNotValid)
{
    OD od;
    const fs::path dir = makeDir("missing_bboxes");
    WriteFrameJson(dir, 0, kLDNetedStage, 1, 999);
    WriteImuCsv(dir, ValidImuRows());
    EXPECT_EQ(od.DatasetPrepare(dir.string(), MakeCalibration()),
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, DatasetPrepareUnknownRegionAndNoValidMeasurementsReturnsODMeasNotValid)
{
    OD od;
    const fs::path dir = makeDir("unknown_region");
    WriteFrameJson(dir, 0, kLDNetedStage, 1, 2, static_cast<int>(RegionID::UNKNOWN));
    WriteImuCsv(dir, ValidImuRows());
    EXPECT_EQ(od.DatasetPrepare(dir.string(), MakeCalibration()),
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, DatasetPrepareClassIdOutOfRangeReturnsODMeasNotValid)
{
    OD od;
    const fs::path dir = makeDir("class_out_of_range");
    WriteFrameJson(dir, 0, kLDNetedStage, 1, 2, kRegion17R, 999999);
    WriteImuCsv(dir, ValidImuRows());
    EXPECT_EQ(od.DatasetPrepare(dir.string(), MakeCalibration()),
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, DatasetPrepareMissingImuCsvReturnsFileDoesNotExist)
{
    OD od;
    const fs::path dir = makeDir("missing_imu_prepare");
    WriteFrameJson(dir, 0, kLDNetedStage, 1);
    EXPECT_EQ(od.DatasetPrepare(dir.string(), MakeCalibration()),
              ErrorCode::FILE_DOES_NOT_EXIST);
}

TEST_F(ODTest, DatasetPrepareMalformedImuCsvReturnsODMeasNotValid)
{
    OD od;
    const fs::path dir = makeDir("malformed_imu_prepare");
    WriteFrameJson(dir, 0, kLDNetedStage, 1);
    WriteImuCsv(dir, {}, true);
    EXPECT_EQ(od.DatasetPrepare(dir.string(), MakeCalibration()),
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, DatasetPrepareMinimalValidDatasetReturnsOKAndWritesCsv)
{
    OD od;
    const fs::path dir = makeDir("valid_prepare");
    WriteFrameJson(dir, 0, kLDNetedStage, 1);
    WriteImuCsv(dir, ValidImuRows());

    EXPECT_EQ(od.DatasetPrepare(dir.string(), MakeCalibration()), ErrorCode::OK);
    EXPECT_TRUE(fs::exists(dir / "landmark_measurements.csv"));

    std::ifstream f(dir / "landmark_measurements.csv");
    ASSERT_TRUE(f.is_open());
    std::string header;
    std::getline(f, header);
    EXPECT_EQ(header, "timestamp_ms,bearing_x,bearing_y,bearing_z,eci_x_km,eci_y_km,eci_z_km,group,sigma");
    std::string row;
    EXPECT_TRUE(static_cast<bool>(std::getline(f, row)));
}

TEST_F(ODTest, LoadODMeasurementsMissingLandmarkCsvReturnsFileDoesNotExist)
{
    const fs::path dir = makeDir("load_missing_lm");
    WriteImuCsv(dir, ValidImuRows());
    EXPECT_EQ(LoadODMeasurementsFromDataset(dir.string()).code,
              ErrorCode::FILE_DOES_NOT_EXIST);
}

TEST_F(ODTest, LoadODMeasurementsMissingImuCsvReturnsFileDoesNotExist)
{
    const fs::path dir = makeDir("load_missing_imu");
    WriteLandmarkCsv(dir, {LandmarkRow(static_cast<double>(kBaseUnixMs), 0)});
    EXPECT_EQ(LoadODMeasurementsFromDataset(dir.string()).code,
              ErrorCode::FILE_DOES_NOT_EXIST);
}

TEST_F(ODTest, LoadODMeasurementsEmptyLandmarkCsvReturnsODMeasNotValid)
{
    const fs::path dir = makeDir("load_empty_lm");
    WriteLandmarkCsv(dir, {});
    WriteImuCsv(dir, ValidImuRows());
    EXPECT_EQ(LoadODMeasurementsFromDataset(dir.string()).code,
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, LoadODMeasurementsMalformedLandmarkRowsReturnsODMeasNotValid)
{
    const fs::path dir = makeDir("load_bad_lm");
    WriteLandmarkCsv(dir, {}, true);
    WriteImuCsv(dir, ValidImuRows());
    EXPECT_EQ(LoadODMeasurementsFromDataset(dir.string()).code,
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, LoadODMeasurementsEmptyImuCsvReturnsODMeasNotValid)
{
    const fs::path dir = makeDir("load_empty_imu");
    WriteLandmarkCsv(dir, {LandmarkRow(static_cast<double>(kBaseUnixMs), 0)});
    WriteImuCsv(dir, {});
    EXPECT_EQ(LoadODMeasurementsFromDataset(dir.string()).code,
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, LoadODMeasurementsMalformedImuRowsReturnsODMeasNotValid)
{
    const fs::path dir = makeDir("load_bad_imu");
    WriteLandmarkCsv(dir, {LandmarkRow(static_cast<double>(kBaseUnixMs), 0)});
    WriteImuCsv(dir, {}, true);
    EXPECT_EQ(LoadODMeasurementsFromDataset(dir.string()).code,
              ErrorCode::ODMEAS_NOT_VALID);
}

TEST_F(ODTest, LoadODMeasurementsValidCsvsReturnsOKAndShapes)
{
    const fs::path dir = makeDir("load_valid");
    WriteLandmarkCsv(dir, {
        LandmarkRow(static_cast<double>(kBaseUnixMs), 0),
        LandmarkRow(static_cast<double>(kBaseUnixMs + 1000), 0),
        LandmarkRow(static_cast<double>(kBaseUnixMs + 2000), 1),
    });
    WriteImuCsv(dir, ValidImuRows());

    const ODMeasurementsResult result = LoadODMeasurementsFromDataset(dir.string());
    ASSERT_EQ(result.code, ErrorCode::OK);
    EXPECT_EQ(result.measurements.landmark_measurements.cols(), LandmarkMeasurementIdx::LANDMARK_COUNT);
    EXPECT_EQ(result.measurements.gyro_measurements.cols(), GyroMeasurementIdx::GYRO_MEAS_COUNT);
    EXPECT_EQ(result.measurements.group_starts.rows(), result.measurements.landmark_measurements.rows());
    EXPECT_EQ(result.measurements.landmark_uncertainties.rows(),
              result.measurements.landmark_measurements.rows());
}

TEST_F(ODTest, LoadODMeasurementsGroupStartsComputedFromGroupColumn)
{
    const fs::path dir = makeDir("load_groups");
    WriteLandmarkCsv(dir, {
        LandmarkRow(static_cast<double>(kBaseUnixMs), 0),
        LandmarkRow(static_cast<double>(kBaseUnixMs + 1000), 0),
        LandmarkRow(static_cast<double>(kBaseUnixMs + 2000), 1),
        LandmarkRow(static_cast<double>(kBaseUnixMs + 3000), 1),
        LandmarkRow(static_cast<double>(kBaseUnixMs + 4000), 3),
    });
    WriteImuCsv(dir, ValidImuRows());

    const ODMeasurementsResult result = LoadODMeasurementsFromDataset(dir.string());
    ASSERT_EQ(result.code, ErrorCode::OK);
    ASSERT_EQ(result.measurements.group_starts.rows(), 5);
    EXPECT_TRUE(result.measurements.group_starts(0));
    EXPECT_FALSE(result.measurements.group_starts(1));
    EXPECT_TRUE(result.measurements.group_starts(2));
    EXPECT_FALSE(result.measurements.group_starts(3));
    EXPECT_TRUE(result.measurements.group_starts(4));
}

TEST_F(ODTest, LoadODMeasurementsConvertsTimestampMsToJ2000)
{
    const fs::path dir = makeDir("load_timestamp");
    WriteLandmarkCsv(dir, {LandmarkRow(static_cast<double>(kBaseUnixMs), 0)});
    WriteImuCsv(dir, ValidImuRows());

    const ODMeasurementsResult result = LoadODMeasurementsFromDataset(dir.string());
    ASSERT_EQ(result.code, ErrorCode::OK);
    const double expected = static_cast<double>(kBaseUnixMs) / 1000.0 -
                            static_cast<double>(J2000_EPOCH_UNIX_S);
    EXPECT_DOUBLE_EQ(result.measurements.landmark_measurements(0, LandmarkMeasurementIdx::LANDMARK_TIMESTAMP),
                     expected);
}

TEST_F(ODTest, LoadODMeasurementsConvertsGyroDpsToRadps)
{
    const fs::path dir = makeDir("load_gyro_units");
    WriteLandmarkCsv(dir, {LandmarkRow(static_cast<double>(kBaseUnixMs), 0)});
    WriteImuCsv(dir, {{{static_cast<double>(kBaseUnixMs), 180.0, -90.0, 45.0}}});

    const ODMeasurementsResult result = LoadODMeasurementsFromDataset(dir.string());
    ASSERT_EQ(result.code, ErrorCode::OK);
    EXPECT_NEAR(result.measurements.gyro_measurements(0, GyroMeasurementIdx::ANG_VEL_X), M_PI, 1e-12);
    EXPECT_NEAR(result.measurements.gyro_measurements(0, GyroMeasurementIdx::ANG_VEL_Y), -M_PI / 2.0, 1e-12);
    EXPECT_NEAR(result.measurements.gyro_measurements(0, GyroMeasurementIdx::ANG_VEL_Z), M_PI / 4.0, 1e-12);
}

TEST_F(ODTest, LoadODMeasurementsSkipsMalformedRowsButLoadsValidRows)
{
    const fs::path dir = makeDir("load_mixed_rows");
    {
        std::ofstream f(dir / "landmark_measurements.csv");
        ASSERT_TRUE(f.is_open());
        f << std::fixed << std::setprecision(9);
        f << "timestamp_ms,bearing_x,bearing_y,bearing_z,eci_x_km,eci_y_km,eci_z_km,group,sigma\n";
        f << "bad,row\n";
        const auto row = LandmarkRow(static_cast<double>(kBaseUnixMs), 0);
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) f << ',';
            f << row[i];
        }
        f << '\n';
    }
    {
        std::ofstream f(dir / "imu_data.csv");
        ASSERT_TRUE(f.is_open());
        f << std::fixed << std::setprecision(9);
        f << "timestamp_ms,gyro_x_dps,gyro_y_dps,gyro_z_dps\n";
        f << "bad,row\n";
        f << kBaseUnixMs << ",0,0,0\n";
    }

    const ODMeasurementsResult result = LoadODMeasurementsFromDataset(dir.string());
    ASSERT_EQ(result.code, ErrorCode::OK);
    EXPECT_EQ(result.measurements.landmark_measurements.rows(), 1);
    EXPECT_EQ(result.measurements.gyro_measurements.rows(), 1);
}

TEST_F(ODTest, RunODOnDatasetMissingDatasetReturnsODMeasNotValidAndDatasetNotAvailable)
{
    ODRequest request;
    request.dataset_folder = (root / "missing").string();
    const ODResult result = RunODOnDataset(request);
    EXPECT_EQ(result.code, ErrorCode::ODMEAS_NOT_VALID);
    EXPECT_EQ(result.stage, ODStage::DATASET_NOT_AVAILABLE);
}

TEST_F(ODTest, RunODOnDatasetUnprocessedDatasetReturnsODMeasNotValidAndDatasetNotProcessed)
{
    ODRequest request;
    request.dataset_folder = makeDir("run_unprocessed").string();
    const ODResult result = RunODOnDataset(request);
    EXPECT_EQ(result.code, ErrorCode::ODMEAS_NOT_VALID);
    EXPECT_EQ(result.stage, ODStage::DATASET_NOT_PROCESSED);
}

TEST_F(ODTest, RunODOnDatasetInvalidConfigPathReturnsFileDoesNotExistAndFailed)
{
    const fs::path dir = makeDir("run_bad_config");
    WriteLandmarkCsv(dir, {});

    ODRequest request;
    request.dataset_folder = dir.string();
    request.od_config_path = (root / "missing_config.toml").string();
    const ODResult result = RunODOnDataset(request);
    EXPECT_EQ(result.code, ErrorCode::FILE_DOES_NOT_EXIST);
    EXPECT_EQ(result.stage, ODStage::FAILED);
}

TEST_F(ODTest, RunODOnDatasetProcessedDatasetPrepareFailsPropagatesPrepareCode)
{
    const fs::path dir = makeDir("run_prepare_fails");
    WriteFrameJson(dir, 0, kLDNetedStage, 1, 2, kRegion17R, 0, true, false);

    ODRequest request;
    request.dataset_folder = dir.string();
    const ODResult result = RunODOnDataset(request);
    EXPECT_EQ(result.code, ErrorCode::ODMEAS_NOT_VALID);
    EXPECT_EQ(result.stage, ODStage::FAILED);
}

TEST_F(ODTest, RunODOnDatasetMeasurementsReadyButLoadFailsPropagatesLoadCode)
{
    const fs::path dir = makeDir("run_load_fails");
    WriteLandmarkCsv(dir, {});

    ODRequest request;
    request.dataset_folder = dir.string();
    const ODResult result = RunODOnDataset(request);
    EXPECT_EQ(result.code, ErrorCode::ODMEAS_NOT_VALID);
    EXPECT_EQ(result.stage, ODStage::FAILED);
}
