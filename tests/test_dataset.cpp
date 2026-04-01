/*
    Test file to replicate what should happen in FSW when the command to start
    dataset collection is received.
*/

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include "toml.hpp"
#include "spdlog/spdlog.h"
#include <functional>

// Expose private members for ProcessFrames and DatasetManager internals.
// Placed after third-party includes to avoid exposing their implementation details.
#define private public

#include "spdlog/spdlog.h"
#include "vision/dataset_manager.hpp"
#include "inference/inference_manager.hpp"
#include "core/timing.hpp"
#include "configuration.hpp"
#include <core/errors.hpp>

#define ERASE_TEST_FILES false

namespace fs = std::filesystem;
using FrameVec = std::vector<std::tuple<uint8_t, uint64_t>>;

// ── Shared helper ─────────────────────────────────────────────────────────────

static Dataset makeDatasetAt(uint64_t start_ms, double period_s)
{
    return Dataset(period_s, 10, CAPTURE_MODE::PERIODIC,
                   IMU_COLLECTION_MODE::GYRO_ONLY, 60, 1.0f,
                   ProcessingStage::NotPrefiltered, start_ms);
}

// ── DatasetTest fixture ───────────────────────────────────────────────────────

struct DatasetTest : ::testing::Test
{
    double              max_period              = 60.0;
    uint8_t             nb_frames               = 100;
    CAPTURE_MODE        capture_mode            = CAPTURE_MODE::PERIODIC;
    IMU_COLLECTION_MODE imu_collection_mode     = IMU_COLLECTION_MODE::GYRO_ONLY;
    uint8_t             image_capture_rate      = 60;
    float               imu_sample_rate_hz      = 1.0f;
    ProcessingStage     target_processing_stage = ProcessingStage::NotPrefiltered;
    uint64_t            capture_start_time      = timing::GetCurrentTimeMs();

    bool isValid() const {
        return Dataset::isValidConfiguration(max_period, nb_frames, capture_mode,
            imu_collection_mode, image_capture_rate, imu_sample_rate_hz,
            target_processing_stage, capture_start_time);
    }

    Dataset makeDataset() const {
        return Dataset(max_period, nb_frames, capture_mode, imu_collection_mode,
                       image_capture_rate, imu_sample_rate_hz,
                       target_processing_stage, capture_start_time);
    }
};

// ── DatasetConfigurationCheck ─────────────────────────────────────────────────

TEST_F(DatasetTest, DatasetConfigurationCheck)
{
    EXPECT_TRUE(isValid());

    auto checkInvalid = [&](auto& field, auto bad_val) {
        auto saved = field; field = bad_val; EXPECT_FALSE(isValid()); field = saved;
    };
    checkInvalid(max_period,         0.0);
    checkInvalid(nb_frames,          uint8_t(0));
    checkInvalid(nb_frames,          uint8_t(256));
    checkInvalid(capture_mode,       CAPTURE_MODE::IDLE);
    checkInvalid(capture_mode,       CAPTURE_MODE::CAPTURE_SINGLE);
    checkInvalid(image_capture_rate, uint8_t(0));
    checkInvalid(imu_sample_rate_hz, 0.0f);

    // TODO: Check with invalid configuration that an std::invalid_argument exception is thrown
    ASSERT_NO_THROW(makeDataset());
    Dataset dataset = makeDataset();

    EXPECT_EQ(dataset.GetCaptureStartTime(),      capture_start_time);
    EXPECT_EQ(dataset.GetTargetFrameNb(),         nb_frames);
    EXPECT_EQ(dataset.GetMaximumPeriod(),         max_period);
    EXPECT_EQ(dataset.GetDatasetCaptureMode(),    capture_mode);
    EXPECT_EQ(dataset.GetIMUCollectionMode(),     imu_collection_mode);
    EXPECT_EQ(dataset.GetImageCaptureRate(),      image_capture_rate);
    EXPECT_EQ(dataset.GetIMUSampleRateHz(),       imu_sample_rate_hz);
    EXPECT_EQ(dataset.GetTargetProcessingStage(), target_processing_stage);

    const std::string folder = "data/datasets/" + std::to_string(capture_start_time) + "/";
    EXPECT_EQ(dataset.GetFolderPath(),  folder);
    EXPECT_EQ(dataset.GetIMUFilePath(), folder + "imu_data.csv");

    // Config file: valid on creation, invalid after corruption
    const std::string config_path = folder + "dataset_config.toml";
    EXPECT_TRUE(Dataset::isValidConfigurationFile(config_path));
    ASSERT_NO_THROW(Dataset{folder});

    toml::table config = toml::parse_file(config_path);
    auto writeConfig = [&] {
        std::ofstream f(config_path, std::ofstream::out | std::ofstream::trunc);
        f << config;
    };
    auto expectBad = [&] {
        EXPECT_FALSE(Dataset::isValidConfigurationFile(config_path));
        EXPECT_THROW(Dataset{folder}, std::invalid_argument);
    };

    // Table-driven: for each field verify (a) missing → invalid, (b) bad value → invalid.
    // insert_good restores the field so the next iteration starts from a valid config.
    struct FieldTest {
        std::string field;
        std::function<void(toml::table&)> insert_bad;
        std::function<void(toml::table&)> insert_good;
    };

    const std::vector<FieldTest> field_tests = {
        {"maximum_period",
            [](toml::table& c){ c.insert("maximum_period",           0.0f); },
            [](toml::table& c){ c.erase("maximum_period");  c.insert("maximum_period",      60.0f); }},
        {"target_frame_nb",
            [](toml::table& c){ c.insert("target_frame_nb",          0); },
            [](toml::table& c){ c.erase("target_frame_nb"); c.insert("target_frame_nb",      100); }},
        {"dataset_capture_mode",
            [](toml::table& c){ c.insert("dataset_capture_mode",     0); },
            [](toml::table& c){ c.erase("dataset_capture_mode"); c.insert("dataset_capture_mode", 2); }},
        {"imu_collection_mode",
            [](toml::table& c){ c.insert("imu_collection_mode",     -1); },
            [](toml::table& c){ c.erase("imu_collection_mode"); c.insert("imu_collection_mode",   2); }},
        {"image_capture_rate",
            [](toml::table& c){ c.insert("image_capture_rate",      -1); },
            [](toml::table& c){ c.erase("image_capture_rate"); c.insert("image_capture_rate",    60); }},
        {"imu_sample_rate_hz",
            [](toml::table& c){ c.insert("imu_sample_rate_hz",      -1); },
            [](toml::table& c){ c.erase("imu_sample_rate_hz"); c.insert("imu_sample_rate_hz", 1.0f); }},
        {"target_processing_stage",
            [](toml::table& c){ c.insert("target_processing_stage", -1); },
            [](toml::table& c){ /* last field — no restore needed */ }},
    };

    for (const auto& [field, insert_bad, insert_good] : field_tests)
    {
        config.erase(field);  writeConfig(); expectBad();
        insert_bad(config);   writeConfig(); expectBad();
        config.erase(field);  insert_good(config);
    }

    if (ERASE_TEST_FILES) fs::remove_all(folder);
}

// ── DatasetStorageAndRetrieval ────────────────────────────────────────────────

TEST_F(DatasetTest, DatasetStorageAndRetrieval)
{
    Dataset dataset = makeDataset();
    const std::string folder = "data/datasets/" + std::to_string(capture_start_time);

    EXPECT_TRUE(fs::exists(folder));
    EXPECT_TRUE(fs::exists(folder + "/dataset_config.toml"));

    Dataset retrieved(folder);
    EXPECT_EQ(retrieved.GetMaximumPeriod(),         60.0);
    EXPECT_EQ(retrieved.GetTargetFrameNb(),         100);
    EXPECT_EQ(retrieved.GetDatasetCaptureMode(),    CAPTURE_MODE::PERIODIC);
    EXPECT_EQ(retrieved.GetCaptureStartTime(),      capture_start_time);
    EXPECT_EQ(retrieved.GetIMUCollectionMode(),     IMU_COLLECTION_MODE::GYRO_ONLY);
    EXPECT_EQ(retrieved.GetImageCaptureRate(),      60);
    EXPECT_EQ(retrieved.GetIMUSampleRateHz(),       1.0f);
    EXPECT_EQ(retrieved.GetTargetProcessingStage(), ProcessingStage::NotPrefiltered);
    ASSERT_THROW(Dataset{"data/datasets/bogus_dataset_path/"}, std::invalid_argument);

    Json j = dataset.toJson();
    for (const char* key : {"folder_path", "capture_start_time", "maximum_period",
                             "target_frame_nb", "dataset_capture_mode", "imu_collection_mode",
                             "image_capture_rate", "imu_sample_rate_hz", "target_processing_stage",
                             "imu_log_file_path", "frame_id_list"})
        EXPECT_TRUE(j.contains(key)) << "missing JSON key: " << key;

    EXPECT_EQ(j["capture_start_time"], capture_start_time);
    EXPECT_EQ(j["maximum_period"],     60.0);
    EXPECT_EQ(j["target_frame_nb"],    100);

    // fromJson round-trip
    Dataset from_json(max_period, nb_frames, capture_mode, imu_collection_mode,
                      image_capture_rate, imu_sample_rate_hz, target_processing_stage,
                      capture_start_time + 100000);
    EXPECT_TRUE(from_json.fromJson(j));
    EXPECT_EQ(from_json.GetCaptureStartTime(), capture_start_time);
    EXPECT_EQ(from_json.GetMaximumPeriod(),    60.0);
    EXPECT_EQ(from_json.GetTargetFrameNb(),    100);
    EXPECT_EQ(from_json.GetFolderPath(),       folder + "/");
}

// ── Dataset::AddStoredFrameIDs ────────────────────────────────────────────────

TEST_F(DatasetTest, AddStoredFrameIDs_Deduplication)
{
    Dataset dataset = makeDataset();

    auto id1 = std::make_tuple(uint8_t(0), uint64_t(100));
    auto id2 = std::make_tuple(uint8_t(1), uint64_t(200));

    dataset.AddStoredFrameID(id1);
    dataset.AddStoredFrameID(id2);
    EXPECT_EQ(dataset.GetStoredFrameIDs().size(), 2u);

    dataset.AddStoredFrameID(id1);
    dataset.AddStoredFrameIDs({id1, id2});
    EXPECT_EQ(dataset.GetStoredFrameIDs().size(), 2u);  // no growth on duplicates

    dataset.AddStoredFrameID(std::make_tuple(uint8_t(0), uint64_t(300)));
    EXPECT_EQ(dataset.GetStoredFrameIDs().size(), 3u);

    if (ERASE_TEST_FILES) fs::remove_all(dataset.GetFolderPath());
}

// ── Dataset::OverlapsWith — parameterized ─────────────────────────────────────
//
// Timestamps are small fixed values (ms ≈ epoch 0) — clearly test data,
// never collide with real capture timestamps.

struct OverlapCase {
    const char* name;
    uint64_t a_start; double a_period;
    uint64_t b_start; double b_period;
    bool expected;
};

class DatasetOverlapTest : public ::testing::TestWithParam<OverlapCase> {};

TEST_P(DatasetOverlapTest, Check)
{
    const auto [name, a_start, a_period, b_start, b_period, expected] = GetParam();
    auto a = makeDatasetAt(a_start, a_period);
    auto b = makeDatasetAt(b_start, b_period);
    EXPECT_EQ(a.OverlapsWith(b), expected);
    EXPECT_EQ(b.OverlapsWith(a), expected);
    fs::remove_all(a.GetFolderPath());
    fs::remove_all(b.GetFolderPath());  // harmless no-op if same folder as a
}

INSTANTIATE_TEST_SUITE_P(
    OverlapCases, DatasetOverlapTest,
    ::testing::Values(
        //                     a_start  a_period  b_start  b_period  expected
        OverlapCase{"NoOverlap",   1000,    10.0,   20000,    10.0,  false},
        OverlapCase{"PartialStart",5000,    10.0,    1000,    10.0,  true },
        OverlapCase{"PartialEnd",  1000,    10.0,    5000,    10.0,  true },
        OverlapCase{"AContainsB",  1000,    60.0,    5000,     5.0,  true },
        OverlapCase{"SameWindow",  1000,    10.0,    1000,    10.0,  true },
        OverlapCase{"Adjacent",    1000,    10.0,   11001,    10.0,  false}
    ),
    [](const ::testing::TestParamInfo<OverlapCase>& i) { return i.param.name; }
);

// ── DatasetProgress ───────────────────────────────────────────────────────────

TEST(DatasetProgressTest, UpdateAccumulation)
{
    DatasetProgress p(10);
    EXPECT_EQ(p.current_frames, 0);
    EXPECT_DOUBLE_EQ(p.completion, 0.0);
    EXPECT_DOUBLE_EQ(p.hit_ratio, 1.0);  // default

    p.Update(3, 0.6);
    EXPECT_EQ(p.current_frames, 3);
    EXPECT_NEAR(p.completion, 0.3, 1e-9);
    EXPECT_NEAR(p.hit_ratio,  0.6, 1e-9);

    p.Update(7, 1.0);
    EXPECT_EQ(p.current_frames, 10);
    EXPECT_NEAR(p.completion, 1.0, 1e-9);
}

TEST(DatasetProgressTest, CumulativeHitRatioAverage)
{
    // Formula: hit_ratio = (new + n_prev * prev) / (n_prev + 1)
    DatasetProgress p(10);
    p.Update(2, 0.4);
    EXPECT_NEAR(p.hit_ratio, 0.4, 1e-9);                       // (0.4 + 0*1.0) / 1

    p.Update(3, 0.8);
    EXPECT_NEAR(p.hit_ratio, 0.6, 1e-9);                       // (0.8 + 1*0.4) / 2

    p.Update(5, 1.0);
    EXPECT_NEAR(p.hit_ratio, (1.0 + 2.0*0.6) / 3.0, 1e-9);    // (1.0 + 2*0.6) / 3
}

// ── DatasetManagerTest fixture ────────────────────────────────────────────────

struct DatasetManagerTest : ::testing::Test
{
    static std::array<CameraConfig, NUM_CAMERAS> makeDummyCamConfigs()
    {
        std::array<CameraConfig, NUM_CAMERAS> c;
        for (int i = 0; i < NUM_CAMERAS; ++i)
            c[i] = {static_cast<int64_t>(i), "/dev/null", 1920, 1080};
        return c;
    }

    InferenceManager                      im;
    std::array<CameraConfig, NUM_CAMERAS> cam_configs { makeDummyCamConfigs() };
    CameraManager                         cam_mgr { cam_configs, im };
    IMUManager                            imu_mgr { IMUConfig{0x00, 0x68, "/dev/null"} };
    std::vector<std::string>              test_folders;

    void TearDown() override
    {
        // Hold a local ref per entry so the DatasetManager destructor does not
        // run while datasets_mtx is locked inside StopDatasetManager.
        for (const auto& key : DatasetManager::ListActiveDatasetManagers()) {
            auto ref = DatasetManager::GetActiveDatasetManager(key);
            DatasetManager::StopDatasetManager(key);
        }
        for (const auto& f : test_folders) fs::remove_all(f);
    }

    std::shared_ptr<DatasetManager> createDM(
        uint64_t           start_ms,
        double             period_s,
        const std::string& key,
        ProcessingStage    target = ProcessingStage::NotPrefiltered,
        CAPTURE_MODE       mode   = CAPTURE_MODE::PERIODIC)
    {
        auto dm = DatasetManager::Create(period_s, 5, mode, start_ms,
                                         IMU_COLLECTION_MODE::GYRO_ONLY, 60, 1.0f,
                                         target, key, cam_mgr, imu_mgr, im);
        test_folders.push_back(dm->current_dataset.GetFolderPath());
        return dm;
    }
};

// ── DatasetManager registry ───────────────────────────────────────────────────

TEST_F(DatasetManagerTest, Registry_CRUD)
{
    auto dm_a = createDM(100000, 10.0, "key_a");
    ASSERT_NE(DatasetManager::GetActiveDatasetManager("key_a"), nullptr);
    EXPECT_EQ(DatasetManager::GetActiveDatasetManager("key_a").get(), dm_a.get());
    EXPECT_EQ(DatasetManager::GetActiveDatasetManager("missing"), nullptr);  // unknown key

    createDM(200000, 10.0, "key_b");
    auto keys = DatasetManager::ListActiveDatasetManagers();
    EXPECT_EQ(keys.size(), 2u);
    EXPECT_NE(std::find(keys.begin(), keys.end(), "key_a"), keys.end());
    EXPECT_NE(std::find(keys.begin(), keys.end(), "key_b"), keys.end());

    // Stop removes from registry (hold local ref to avoid destructor under lock)
    { auto ref = DatasetManager::GetActiveDatasetManager("key_a");
      DatasetManager::StopDatasetManager("key_a"); }
    EXPECT_EQ(DatasetManager::GetActiveDatasetManager("key_a"), nullptr);
    EXPECT_EQ(DatasetManager::ListActiveDatasetManagers().size(), 1u);
}

TEST_F(DatasetManagerTest, Registry_DefaultKeyUsesCreatedAt)
{
    auto dm = createDM(300000, 10.0, DEFAULT_DS_KEY);
    auto keys = DatasetManager::ListActiveDatasetManagers();
    ASSERT_EQ(keys.size(), 1u);
    EXPECT_EQ(keys[0], std::to_string(dm->created_at));
}

// ── DatasetManager overlap rejection ─────────────────────────────────────────

TEST_F(DatasetManagerTest, OverlapRejection)
{
    createDM(700000, 60.0, "base");  // window [700000, 760000]

    // Dataset constructor creates the folder before the overlap check throws.
    test_folders.push_back("data/datasets/710000/");
    EXPECT_THROW(
        DatasetManager::Create(10.0, 5, CAPTURE_MODE::PERIODIC, 710000,
                               IMU_COLLECTION_MODE::GYRO_ONLY, 60, 1.0f,
                               ProcessingStage::NotPrefiltered, "overlap",
                               cam_mgr, imu_mgr, im),
        std::invalid_argument);

    EXPECT_NO_THROW(createDM(820000, 10.0, "after"));  // clear of base window
}

// ── DatasetManager termination check ─────────────────────────────────────────

TEST_F(DatasetManagerTest, IsCompleted)
{
    uint64_t future = timing::GetCurrentTimeMs() + 60000;
    auto dm = createDM(future, 60.0, "pending");
    EXPECT_FALSE(dm->IsCompleted());

    dm->progress.Update(dm->current_dataset.GetTargetFrameNb(), 1.0);
    EXPECT_TRUE(dm->IsCompleted());  // frame count met

    EXPECT_TRUE(createDM(1, 0.1, "expired")->IsCompleted());  // period elapsed
}

// ── DatasetManager::ProcessFrames ────────────────────────────────────────────

TEST_F(DatasetManagerTest, ProcessFrames_EarlyExit)
{
    // target == capture_stage (both NotPrefiltered) → returns without touching processed
    auto dm = createDM(900000, 60.0, "early_exit");
    FrameVec frame_ids = {{uint8_t(0), uint64_t(1234)}};
    FrameVec processed;
    dm->ProcessFrames(frame_ids, processed);
    EXPECT_TRUE(processed.empty());
}

TEST_F(DatasetManagerTest, ProcessFrames_SkipAlreadyProcessed)
{
    auto dm = createDM(950000, 60.0, "skip_proc", ProcessingStage::Prefiltered);
    auto id = std::make_tuple(uint8_t(0), uint64_t(9999));
    FrameVec processed = {id};  // pre-populated
    dm->ProcessFrames({id}, processed);
    EXPECT_EQ(processed.size(), 1u);  // no change — frame was skipped
}

TEST_F(DatasetManagerTest, ProcessFrames_MissingFile_MarkedProcessed)
{
    // File doesn't exist → frame is pushed to processed (fail-safe, no retry)
    auto dm = createDM(960000, 60.0, "missing_file", ProcessingStage::Prefiltered);
    auto id = std::make_tuple(uint8_t(0), uint64_t(77777));
    FrameVec processed;
    dm->ProcessFrames({id}, processed);
    ASSERT_EQ(processed.size(), 1u);
    EXPECT_EQ(processed[0], id);
}

TEST_F(DatasetManagerTest, ProcessFrames_BlackImageRejectedByPrefilter)
{
    // Write a valid black PNG; RunPrefiltering() should reject it as not earth-facing.
    // Either way the frame must end up in processed — never silently dropped.
    auto dm = createDM(970000, 60.0, "prefilter", ProcessingStage::Prefiltered);
    const uint8_t  cam_id    = 0;
    const uint64_t timestamp = 55555;
    const std::string img_path = dm->current_dataset.GetFolderPath()
                                 + "raw_" + std::to_string(timestamp)
                                 + "_"   + std::to_string(cam_id) + ".png";

    cv::Mat black(32, 32, CV_8UC3, cv::Scalar(0, 0, 0));
    ASSERT_TRUE(cv::imwrite(img_path, black)) << "cv::imwrite failed: " << img_path;

    auto id = std::make_tuple(cam_id, timestamp);
    FrameVec processed;
    dm->ProcessFrames({id}, processed);
    ASSERT_EQ(processed.size(), 1u);
    EXPECT_EQ(processed[0], id);
}

// ── Stubs for future tests ────────────────────────────────────────────────────

TEST_F(DatasetTest, DatasetManagerConfigurationCheck)
{
    // TODO: Test error handling of definition of two overlapping datasets
    // TODO: Test configuration of two sequential non-overlapping datasets
}

TEST_F(DatasetTest, DatasetReprocessing)
{
    // TODO: Test reprocessing a dataset
}

// TODO: Integration tests
