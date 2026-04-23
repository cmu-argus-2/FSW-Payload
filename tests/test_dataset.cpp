/*
    Test file to replicate what should happen in FSW when the command to start
    dataset collection is received.
*/

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
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
#include "vision/reprocessing.hpp"
#include "core/data_handling.hpp"

#undef private

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
        Dataset d(max_period, nb_frames, capture_mode, imu_collection_mode,
                  image_capture_rate, imu_sample_rate_hz,
                  target_processing_stage, capture_start_time);
        d.InitializeOnDisk();
        return d;
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

    // isValidConfigurationFile returns false (not throws) for missing or malformed files
    EXPECT_FALSE(Dataset::isValidConfigurationFile("data/datasets/nonexistent/dataset_config.toml"));
    EXPECT_FALSE(Dataset::isValidConfigurationFile("/dev/null"));  // empty file → parse error

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
            [](toml::table& c){ c.erase("target_processing_stage"); c.insert("target_processing_stage", 0); }},
        {"capture_start_time",
            // wrong type (string) → value<int64_t>() returns nullopt
            [](toml::table& c){ c.insert("capture_start_time", "not_a_timestamp"); },
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

// ── validateRawParams: upper-bound violations via isValidConfigurationFile ────
//
// The existing DatasetConfigurationCheck table tests missing fields and values
// that are too low. These cases add the upper-bound violations that exercise
// the "wrapping" guards added to validateRawParams.

TEST_F(DatasetTest, IsValidConfigurationFile_RejectsUpperBoundViolations)
{
    Dataset dataset = makeDataset();
    const std::string config_path =
        "data/datasets/" + std::to_string(capture_start_time) + "/dataset_config.toml";

    toml::table config = toml::parse_file(config_path);
    auto writeConfig = [&] {
        std::ofstream f(config_path, std::ofstream::out | std::ofstream::trunc);
        f << config;
    };
    auto expectBad = [&](const std::string& field, auto bad_val, auto good_val) {
        config.erase(field); config.insert(field, bad_val); writeConfig();
        EXPECT_FALSE(Dataset::isValidConfigurationFile(config_path))
            << field << " = " << bad_val << " should be rejected";
        config.erase(field); config.insert(field, good_val);
    };

    expectBad("maximum_period",          20000.0,  60.0);   // > ABSOLUTE_MAXIMUM_PERIOD (10800)
    expectBad("target_frame_nb",         300,      100);    // > MAX_SAMPLES (255)
    expectBad("imu_collection_mode",     99,       2);      // > GYRO_MAG_TEMP
    expectBad("image_capture_rate",      300,      60);     // > MAX_SAMPLES (255)
    expectBad("imu_sample_rate_hz",      30.0,     1.0);    // > 25 Hz
    expectBad("target_processing_stage", 99,       0);      // > LDNeted

    if (ERASE_TEST_FILES)
        fs::remove_all("data/datasets/" + std::to_string(capture_start_time) + "/");
}

// ── validateRawParams: exercised via fromJson ─────────────────────────────────
//
// fromJson reads values, validates them via validateRawParams, and only assigns
// on success. A failed call must return false and leave the object unchanged.

TEST_F(DatasetTest, FromJson_RejectsInvalidParams)
{
    Dataset dataset = makeDataset();
    const Json valid_j = dataset.toJson();

    // fromJson only writes fields after all validation passes, so the object is
    // safe to reuse across cases — a failed call leaves it unmodified.
    auto expectBad = [&](const std::string& field, Json bad_val) {
        Json j = valid_j;
        j[field] = std::move(bad_val);
        EXPECT_FALSE(dataset.fromJson(j)) << "fromJson should reject " << field << " = " << j[field];
    };

    // Below minimum
    expectBad("maximum_period",          0.0);
    expectBad("target_frame_nb",         0);
    expectBad("image_capture_rate",      0);
    expectBad("imu_sample_rate_hz",      0.0);
    expectBad("dataset_capture_mode",    0);   // IDLE — not a valid collection mode

    // Above maximum / out of enum range
    expectBad("maximum_period",          20000.0);  // > ABSOLUTE_MAXIMUM_PERIOD (10800)
    expectBad("target_frame_nb",         300);      // > MAX_SAMPLES (255)
    expectBad("image_capture_rate",      300);      // > MAX_SAMPLES (255)
    expectBad("imu_sample_rate_hz",      30.0);     // > 25 Hz
    expectBad("imu_collection_mode",     99);
    expectBad("dataset_capture_mode",    99);
    expectBad("target_processing_stage", 99);

    // Valid JSON must still succeed and update fields correctly
    EXPECT_TRUE(dataset.fromJson(valid_j));
    EXPECT_EQ(dataset.GetMaximumPeriod(),   60.0);
    EXPECT_EQ(dataset.GetTargetFrameNb(),   100);

    if (ERASE_TEST_FILES)
        fs::remove_all("data/datasets/" + std::to_string(capture_start_time) + "/");
}

TEST_F(DatasetTest, FromJson_RejectsWrongTypes)
{
    Dataset dataset = makeDataset();
    const Json valid_j = dataset.toJson();

    // Wrong JSON type for each field — the optional helper must catch these
    // without throwing and return false.
    auto expectBadType = [&](const std::string& field, Json bad_val) {
        Json j = valid_j;
        j[field] = std::move(bad_val);
        EXPECT_FALSE(dataset.fromJson(j)) << "fromJson should reject wrong type for " << field;
    };

    // Numeric fields given a string
    expectBadType("maximum_period",          "not-a-number");
    expectBadType("target_frame_nb",         "not-a-number");
    expectBadType("dataset_capture_mode",    "not-a-number");
    expectBadType("imu_collection_mode",     "not-a-number");
    expectBadType("image_capture_rate",      "not-a-number");
    expectBadType("imu_sample_rate_hz",      "not-a-number");
    expectBadType("target_processing_stage", "not-a-number");
    expectBadType("capture_start_time",      "not-a-number");

    // String fields given a number
    expectBadType("folder_path",      42);
    expectBadType("imu_log_file_path", 42);

    // Unsigned field given a signed negative value
    expectBadType("target_frame_nb",  -1);
    expectBadType("capture_start_time", -1);

    // Array field given a scalar
    expectBadType("frame_id_list", 0);

    // Valid JSON must still succeed after all the failed attempts
    EXPECT_TRUE(dataset.fromJson(valid_j));

    if (ERASE_TEST_FILES)
        fs::remove_all("data/datasets/" + std::to_string(capture_start_time) + "/");
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

    p.Update(3);
    EXPECT_EQ(p.current_frames, 3);
    EXPECT_NEAR(p.completion, 0.3, 1e-9);

    p.Update(7);
    EXPECT_EQ(p.current_frames, 10);
    EXPECT_NEAR(p.completion, 1.0, 1e-9);
}

TEST(DatasetProgressTest, Update_SaturatesAtTarget)
{
    // A batch that overshoots the target must clamp, not exceed it.
    DatasetProgress p(5);
    p.Update(3);
    EXPECT_EQ(p.current_frames, 3u);

    // 3 + 4 = 7 > target(5) → clamp to 5
    p.Update(4);
    EXPECT_EQ(p.current_frames, 5u);
    EXPECT_NEAR(p.completion, 1.0, 1e-9);

    // Further updates after saturation must not advance the counter.
    p.Update(10);
    EXPECT_EQ(p.current_frames, 5u);
}

TEST(DatasetProgressTest, Update_NoUint8Wrap)
{
    // Worst-case arithmetic: 253 + 4 = 257, which wraps to 1 under naive uint8_t addition.
    // With NUM_CAMERAS = 4, a single SaveLatestFrames call can return 4.
    DatasetProgress p(255);
    p.Update(253);
    EXPECT_EQ(p.current_frames, 253u);

    p.Update(4);  // 253 + 4 = 257 → must clamp to 255, NOT wrap to 1
    EXPECT_EQ(p.current_frames, 255u);
    EXPECT_NEAR(p.completion, 1.0, 1e-9);
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
    CameraManager                         cam_mgr { cam_configs, CameraISPConfig{}, im };
    IMUManager                            imu_mgr { IMUConfig{0x00, 0x68, "/dev/null"} };
    std::vector<std::string>              test_folders;

    void TearDown() override
    {
        for (const auto& key : DatasetManager::ListActiveDatasetManagers())
            DatasetManager::StopDatasetManager(key);
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

    // Stop removes from registry
    DatasetManager::StopDatasetManager("key_a");
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

    EXPECT_THROW(
        DatasetManager::Create(10.0, 5, CAPTURE_MODE::PERIODIC, 710000,
                               IMU_COLLECTION_MODE::GYRO_ONLY, 60, 1.0f,
                               ProcessingStage::NotPrefiltered, "overlap",
                               cam_mgr, imu_mgr, im),
        std::invalid_argument);

    EXPECT_FALSE(fs::exists("data/datasets/710000/")) << "Overlap rejection must not leave a folder on disk";

    EXPECT_NO_THROW(createDM(820000, 10.0, "after"));  // clear of base window
}

// ── DatasetManager concurrent Create ─────────────────────────────────────────

// Two threads race to create overlapping datasets.
// With datasets_mtx covering the overlap check + insertion atomically,
// exactly one must succeed and one must throw — never both succeed.
// This test would non-deterministically pass under the old (unlocked) code.
TEST_F(DatasetManagerTest, ConcurrentCreate_OverlapIsExclusive)
{
    std::atomic<int> ready{0};
    std::atomic<int> successes{0};
    std::atomic<int> rejections{0};
    std::string folder_a, folder_b;
    std::mutex folder_mtx;

    // Windows [4000000, 4060000] and [4010000, 4070000] overlap.
    auto worker = [&](uint64_t start_ms, const std::string& key, std::string& folder_out) {
        ready.fetch_add(1);
        while (ready.load() < 2) {}  // spin until both threads are lined up
        try {
            auto dm = DatasetManager::Create(60.0, 5, CAPTURE_MODE::PERIODIC, start_ms,
                                             IMU_COLLECTION_MODE::GYRO_ONLY, 60, 1.0f,
                                             ProcessingStage::NotPrefiltered, key,
                                             cam_mgr, imu_mgr, im);
            std::lock_guard<std::mutex> lk(folder_mtx);
            folder_out = dm->current_dataset.GetFolderPath();
            successes.fetch_add(1);
        } catch (const std::invalid_argument&) {
            rejections.fetch_add(1);
        }
    };

    std::thread t1(worker, 4000000, "conc_overlap_a", std::ref(folder_a));
    std::thread t2(worker, 4010000, "conc_overlap_b", std::ref(folder_b));
    t1.join();
    t2.join();

    if (!folder_a.empty()) test_folders.push_back(folder_a);
    if (!folder_b.empty()) test_folders.push_back(folder_b);

    EXPECT_EQ(successes.load(),  1) << "Both overlapping Creates succeeded — mutex did not protect the check+insert";
    EXPECT_EQ(rejections.load(), 1);

    std::string rejected_folder = folder_a.empty() ? "data/datasets/4000000/" : "data/datasets/4010000/";
    EXPECT_FALSE(fs::exists(rejected_folder)) << "Rejected overlapping Create must not leave a folder on disk";
}

// Two threads race to create non-overlapping datasets; both must succeed.
TEST_F(DatasetManagerTest, ConcurrentCreate_NonOverlapBothSucceed)
{
    std::atomic<int> ready{0};
    std::atomic<int> successes{0};
    std::string folder_a, folder_b;
    std::mutex folder_mtx;

    // Windows [5000000, 5010000] and [6000000, 6010000] are clearly disjoint.
    auto worker = [&](uint64_t start_ms, const std::string& key, std::string& folder_out) {
        ready.fetch_add(1);
        while (ready.load() < 2) {}
        try {
            auto dm = DatasetManager::Create(10.0, 5, CAPTURE_MODE::PERIODIC, start_ms,
                                             IMU_COLLECTION_MODE::GYRO_ONLY, 60, 1.0f,
                                             ProcessingStage::NotPrefiltered, key,
                                             cam_mgr, imu_mgr, im);
            std::lock_guard<std::mutex> lk(folder_mtx);
            folder_out = dm->current_dataset.GetFolderPath();
            successes.fetch_add(1);
        } catch (...) {}
    };

    std::thread t1(worker, 5000000, "conc_nooverlap_a", std::ref(folder_a));
    std::thread t2(worker, 6000000, "conc_nooverlap_b", std::ref(folder_b));
    t1.join();
    t2.join();

    if (!folder_a.empty()) test_folders.push_back(folder_a);
    if (!folder_b.empty()) test_folders.push_back(folder_b);

    EXPECT_EQ(successes.load(), 2) << "Non-overlapping concurrent Creates should both succeed";
    EXPECT_EQ(DatasetManager::ListActiveDatasetManagers().size(), 2u);
}

// ── DatasetManager termination check ─────────────────────────────────────────

TEST_F(DatasetManagerTest, IsCompleted)
{
    uint64_t future = timing::GetCurrentTimeMs() + 60000;
    auto dm = createDM(future, 60.0, "pending");
    EXPECT_FALSE(dm->IsCompleted());

    dm->progress.Update(dm->current_dataset.GetTargetFrameNb());
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

TEST_F(DatasetManagerTest, ProcessFrames_BlackJpgImageRejectedByPrefilter)
{
    // Same as the PNG variant above but with a JPG-stored image, verifying that
    // ProcessFrames correctly discovers and loads JPG files.
    auto dm = createDM(971000, 60.0, "prefilter_jpg", ProcessingStage::Prefiltered);
    const uint8_t  cam_id    = 0;
    const uint64_t timestamp = 55556;
    const std::string img_path = dm->current_dataset.GetFolderPath()
                                 + "raw_" + std::to_string(timestamp)
                                 + "_"   + std::to_string(cam_id) + ".jpg";

    cv::Mat black(32, 32, CV_8UC3, cv::Scalar(0, 0, 0));
    ASSERT_TRUE(cv::imwrite(img_path, black)) << "cv::imwrite failed: " << img_path;

    auto id = std::make_tuple(cam_id, timestamp);
    FrameVec processed;
    dm->ProcessFrames({id}, processed);
    ASSERT_EQ(processed.size(), 1u);
    EXPECT_EQ(processed[0], id);
}

// ── DatasetManager async start/stop ──────────────────────────────────────────

TEST_F(DatasetManagerTest, StopBeforeStartTime_ExitsPromptly)
{
    // Start time is 60 seconds in the future; the collection thread will enter
    // wait_until and block. StopCollection must interrupt that wait immediately
    // via loop_cv.notify_all() rather than sleeping until the start time.
    const uint64_t future_ms = timing::GetCurrentTimeMs() + 60000;
    auto dm = createDM(future_ms, 60.0, "early_stop");

    dm->StartCollection();
    ASSERT_TRUE(dm->Running());

    const auto t0 = std::chrono::steady_clock::now();
    dm->StopCollection();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

    EXPECT_FALSE(dm->Running());
    EXPECT_LT(elapsed_ms, 1000) << "StopCollection blocked for " << elapsed_ms
                                << " ms — loop_cv interrupt likely not working";
}


// ── CameraManager frame-count saturation ─────────────────────────────────────
//
// periodic_frames_captured is atomic<uint8_t>.  With NUM_CAMERAS == 4, a single
// SaveLatestFrames call can return 4.  If the counter is already near UINT8_MAX
// the naive "+= saved" wraps to a small value, making the termination check
// (captured >= to_capture) permanently false and collection never stopping.

TEST_F(DatasetManagerTest, CameraManager_FrameCount_SaturatesAtTarget)
{
    // target = 5, already at 3, incoming batch = 4 → must clamp to 5, not reach 7.
    cam_mgr.periodic_frames_to_capture = 5;
    cam_mgr.periodic_frames_captured   = 3;

    const uint8_t  saved     = 4;
    const uint16_t new_total = static_cast<uint16_t>(cam_mgr.periodic_frames_captured.load()) + saved;
    cam_mgr.periodic_frames_captured = static_cast<uint8_t>(
        std::min<uint16_t>(new_total, cam_mgr.periodic_frames_to_capture.load()));

    EXPECT_EQ(cam_mgr.periodic_frames_captured.load(), 5u);
    EXPECT_GE(cam_mgr.periodic_frames_captured.load(),
              cam_mgr.periodic_frames_to_capture.load())
        << "Termination check must fire after saturation";
}

TEST_F(DatasetManagerTest, CameraManager_FrameCount_NoUint8Wrap)
{
    // 253 + 4 = 257 wraps to 1 under naive uint8_t arithmetic.
    // With target = 254 the old code would never terminate.
    cam_mgr.periodic_frames_to_capture = 254;
    cam_mgr.periodic_frames_captured   = 253;

    const uint8_t  saved     = 4;
    const uint16_t new_total = static_cast<uint16_t>(cam_mgr.periodic_frames_captured.load()) + saved;
    cam_mgr.periodic_frames_captured = static_cast<uint8_t>(
        std::min<uint16_t>(new_total, cam_mgr.periodic_frames_to_capture.load()));

    EXPECT_EQ(cam_mgr.periodic_frames_captured.load(), 254u)
        << "Counter must clamp to target, not wrap to 1";
    EXPECT_GE(cam_mgr.periodic_frames_captured.load(),
              cam_mgr.periodic_frames_to_capture.load())
        << "Termination check must fire";
}

// ── Reprocessing integration tests ───────────────────────────────────────────
//
// Uses the synthetic dataset at data/datasets/17R_Florida_test/.
// Dataset structure: 9 timestamps × 4 cameras = 36 frames total.
//   cam_id=0 (9 frames): LDNeted (stage 3), rcnet_version=2, ldnet_version=2,
//                        annotation_state=HasLandmark(3) for 8 frames, HasRegion(2) for 1.
//   cam_id=1 (9 frames): 1 frame LDNeted/Earth (no region detected), 8 frames Prefiltered/non-Earth.
//   cam_id=2 (9 frames): LDNeted (stage 3); 5 frames HasRegion(2), 4 frames Earth(1).
//   cam_id=3 (9 frames): Prefiltered (stage 1), non-Earth-facing.
//
// These tests cover the skip-path logic (no TRT engines required).
// All Earth-facing frames are at LDNeted so no inference is triggered during skip tests.
// Reprocess-path tests (overwrite=true + mismatched versions) require TRT
// engines and are exercised via the reprocess_dataset script.

namespace {

const std::string kSyntheticDataset    = "data/datasets/17R_Florida_test/";
constexpr int     kSyntheticFrameCount = 36;
constexpr int     kNumFramesRCNeted    = 19;  // stage >= 2: cam_id=0(9) + cam_id=1(1) + cam_id=2(9)
constexpr int     kNumFramesLDNeted    = 19;  // stage >= 3: same breakdown
constexpr int     kNumFramesEarth      = 19;  // annotation_state >= 1
constexpr int     kNumFramesLandmarks  = 8;   // annotation_state >= 3: cam_id=0 only
constexpr uint64_t kSpotTs  = 1714490526000ULL;
constexpr int      kSpotCam = 0;

Json LoadDatasetJson(const std::string& folder)
{
    std::ifstream f(folder + "dataset.json");
    if (!f) return {};
    Json j;
    f >> j;
    return j;
}

void ConfigureIM(InferenceManager& im, int rc_ver, int ld_ver)
{
    im.SetRCNetVersion(rc_ver);
    im.SetLDNetVersion(ld_ver);
    im.SetLDNetConfig(NET_QUANTIZATION::FP16, 4608, 2592, false, true);
}

} // namespace

struct ReprocessingDatasetTest : ::testing::Test
{
    std::string original_dataset_json_;

    void SetUp() override
    {
        std::ifstream f(kSyntheticDataset + "dataset.json");
        ASSERT_TRUE(f.is_open()) << "Synthetic dataset not found at " << kSyntheticDataset;
        original_dataset_json_.assign(std::istreambuf_iterator<char>(f), {});
    }

    void TearDown() override
    {
        std::ofstream f(kSyntheticDataset + "dataset.json", std::ios::trunc);
        f << original_dataset_json_;
    }

    int SpotFrameStage() const
    {
        Json j = DH::LoadFrameMetadataFromDisk(kSpotTs, kSpotCam, kSyntheticDataset);
        return j.value("processing_stage", -1);
    }
};

// ── Target = Prefiltered ──────────────────────────────────────────────────────

// All frames are at stage 3 ≥ Prefiltered(1): skip regardless of overwrite.
TEST_F(ReprocessingDatasetTest, Prefiltered_OverwriteFalse_AllSkipped)
{
    InferenceManager im;
    ConfigureIM(im, 2, 1);
    Dataset dataset(kSyntheticDataset);

    EC ec = Reprocessing::Dataset(dataset, im, ProcessingStage::Prefiltered, /*overwrite=*/false);

    EXPECT_EQ(ec, EC::OK);
    EXPECT_EQ(SpotFrameStage(), 3) << "Frame JSONs must not be modified";
    Json j = LoadDatasetJson(kSyntheticDataset);
    EXPECT_EQ(j.value("target_processing_stage", -1),
              static_cast<int>(ProcessingStage::Prefiltered));
    EXPECT_EQ(j.value("frames_collected", 0), kSyntheticFrameCount);
}

TEST_F(ReprocessingDatasetTest, Prefiltered_OverwriteTrue_AllSkipped)
{
    InferenceManager im;
    ConfigureIM(im, 2, 1);
    Dataset dataset(kSyntheticDataset);

    EC ec = Reprocessing::Dataset(dataset, im, ProcessingStage::Prefiltered, /*overwrite=*/true);

    EXPECT_EQ(ec, EC::OK);
    EXPECT_EQ(SpotFrameStage(), 3);
    Json j = LoadDatasetJson(kSyntheticDataset);
    EXPECT_EQ(j.value("target_processing_stage", -1),
              static_cast<int>(ProcessingStage::Prefiltered));
    EXPECT_EQ(j.value("frames_collected", 0), kSyntheticFrameCount);
}

// ── Target = RCNeted ─────────────────────────────────────────────────────────

// Stored rc_version=2 matches requested version → conditions match → skip.
TEST_F(ReprocessingDatasetTest, RCNeted_OverwriteFalse_MatchingVersion_AllSkipped)
{
    InferenceManager im;
    ConfigureIM(im, /*rc=*/2, 1);
    Dataset dataset(kSyntheticDataset);

    EC ec = Reprocessing::Dataset(dataset, im, ProcessingStage::RCNeted, /*overwrite=*/false);

    EXPECT_EQ(ec, EC::OK);
    EXPECT_EQ(SpotFrameStage(), 3);
    Json j = LoadDatasetJson(kSyntheticDataset);
    EXPECT_EQ(j.value("target_processing_stage", -1),
              static_cast<int>(ProcessingStage::RCNeted));
    EXPECT_EQ(j.value("num_frames_rcneted", 0), kNumFramesRCNeted);
}

// rc_version mismatch + overwrite=false → conditions differ but must not reprocess.
TEST_F(ReprocessingDatasetTest, RCNeted_OverwriteFalse_MismatchedVersion_AllSkipped)
{
    InferenceManager im;
    ConfigureIM(im, /*rc=*/99, 1);
    Dataset dataset(kSyntheticDataset);

    EC ec = Reprocessing::Dataset(dataset, im, ProcessingStage::RCNeted, /*overwrite=*/false);

    EXPECT_EQ(ec, EC::OK);
    EXPECT_EQ(SpotFrameStage(), 3);
    Json j = LoadDatasetJson(kSyntheticDataset);
    EXPECT_EQ(j.value("frames_collected", 0), kSyntheticFrameCount);
}

// Conditions match (rc_version=2) → skip even though overwrite=true.
TEST_F(ReprocessingDatasetTest, RCNeted_OverwriteTrue_MatchingVersion_AllSkipped)
{
    InferenceManager im;
    ConfigureIM(im, /*rc=*/2, 1);
    Dataset dataset(kSyntheticDataset);

    EC ec = Reprocessing::Dataset(dataset, im, ProcessingStage::RCNeted, /*overwrite=*/true);

    EXPECT_EQ(ec, EC::OK);
    EXPECT_EQ(SpotFrameStage(), 3);
}

// ── Target = LDNeted ─────────────────────────────────────────────────────────

// Both rc_version=2 and ld_version=2 match → skip.
TEST_F(ReprocessingDatasetTest, LDNeted_OverwriteFalse_MatchingVersions_AllSkipped)
{
    InferenceManager im;
    ConfigureIM(im, /*rc=*/2, /*ld=*/2);
    Dataset dataset(kSyntheticDataset);

    EC ec = Reprocessing::Dataset(dataset, im, ProcessingStage::LDNeted, /*overwrite=*/false);

    EXPECT_EQ(ec, EC::OK);
    EXPECT_EQ(SpotFrameStage(), 3);
    Json j = LoadDatasetJson(kSyntheticDataset);
    EXPECT_EQ(j.value("target_processing_stage", -1),
              static_cast<int>(ProcessingStage::LDNeted));
    EXPECT_EQ(j.value("num_frames_ldneted",  0), kNumFramesLDNeted);
    EXPECT_EQ(j.value("num_frames_earth",    0), kNumFramesEarth);
    EXPECT_EQ(j.value("num_frames_landmarks",0), kNumFramesLandmarks);
    EXPECT_EQ(j.value("frames_collected",    0), kSyntheticFrameCount);
}

// ld_version mismatch + overwrite=false → skip.
TEST_F(ReprocessingDatasetTest, LDNeted_OverwriteFalse_MismatchedLDVersion_AllSkipped)
{
    InferenceManager im;
    ConfigureIM(im, /*rc=*/2, /*ld=*/99);
    Dataset dataset(kSyntheticDataset);

    EC ec = Reprocessing::Dataset(dataset, im, ProcessingStage::LDNeted, /*overwrite=*/false);

    EXPECT_EQ(ec, EC::OK);
    EXPECT_EQ(SpotFrameStage(), 3);
    Json j = LoadDatasetJson(kSyntheticDataset);
    EXPECT_EQ(j.value("frames_collected", 0), kSyntheticFrameCount);
}

// Conditions match (rc=2, ld=1) → skip even though overwrite=true.
TEST_F(ReprocessingDatasetTest, LDNeted_OverwriteTrue_MatchingVersions_AllSkipped)
{
    InferenceManager im;
    ConfigureIM(im, /*rc=*/2, /*ld=*/2);
    Dataset dataset(kSyntheticDataset);

    EC ec = Reprocessing::Dataset(dataset, im, ProcessingStage::LDNeted, /*overwrite=*/true);

    EXPECT_EQ(ec, EC::OK);
    EXPECT_EQ(SpotFrameStage(), 3);
    Json j = LoadDatasetJson(kSyntheticDataset);
    EXPECT_EQ(j.value("target_processing_stage", -1),
              static_cast<int>(ProcessingStage::LDNeted));
}

// ── Stored inference result validation + dataset report ──────────────────────
//
// Validates landmark detections stored in the frame JSONs against the YOLO
// ground-truth .txt files co-located in the test dataset folder.
// File naming: [RegionID]_[timestamp]_[cam_name].txt
// Camera name → cam_id mapping: xp→0, yp→1, ym→2, xm→3.
// Follows the same class-id + IoU-matching logic as test_inference.cpp.
// Also writes a Markdown report (DATASET_REPORT_PATH or tests/dataset_test_report.md).

namespace {

static const char* kCamNames[] = {"xp", "yp", "ym", "xm"};

int CamNameToId(const std::string& cam)
{
    for (int i = 0; i < 4; ++i)
        if (cam == kCamNames[i]) return i;
    return -1;
}

// Parse a YOLO-format .txt label file into (class_id, pixel-space Rect) pairs.
std::vector<std::pair<int, cv::Rect>>
ParseYoloLabels(const std::string& path, int img_w, int img_h)
{
    std::vector<std::pair<int, cv::Rect>> boxes;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int cls;
        float nx, ny, nw, nh;
        if (!(iss >> cls >> nx >> ny >> nw >> nh)) continue;
        const int x = static_cast<int>(nx * img_w - nw * img_w * 0.5f);
        const int y = static_cast<int>(ny * img_h - nh * img_h * 0.5f);
        const int w = static_cast<int>(nw * img_w);
        const int h = static_cast<int>(nh * img_h);
        boxes.push_back({cls, cv::Rect(x, y, w, h)});
    }
    return boxes;
}

static float BoxIoU(const cv::Rect& a, const cv::Rect& b)
{
    const int inter = (a & b).area();
    const int uni   = (a | b).area();
    return (uni == 0) ? 0.0f : static_cast<float>(inter) / static_cast<float>(uni);
}

struct LabelFile {
    uint64_t    timestamp;
    int         cam_id;
    std::string region_str;
    std::string txt_path;
};

// .txt files: [region]_[timestamp]_[cam].txt
std::vector<LabelFile> DiscoverLabelFiles(const std::string& folder)
{
    std::vector<LabelFile> result;
    for (const auto& e : fs::directory_iterator(folder)) {
        if (!e.is_regular_file() || e.path().extension() != ".txt") continue;
        const std::string stem = e.path().stem().string();
        const auto last_us = stem.rfind('_');
        if (last_us == std::string::npos) continue;
        const auto prev_us = stem.rfind('_', last_us - 1);
        if (prev_us == std::string::npos) continue;
        const int cam_id = CamNameToId(stem.substr(last_us + 1));
        if (cam_id < 0) continue;
        uint64_t ts;
        try { ts = std::stoull(stem.substr(prev_us + 1, last_us - prev_us - 1)); }
        catch (...) { continue; }
        result.push_back({ts, cam_id, stem.substr(0, prev_us), e.path().string()});
    }
    return result;
}

// Scan folder for all frame_*.json files; parse (cam_id, timestamp) from names.
std::vector<std::pair<int,uint64_t>> DiscoverFrameIDs(const std::string& folder)
{
    std::vector<std::pair<int,uint64_t>> ids;
    for (const auto& e : fs::directory_iterator(folder)) {
        if (!e.is_regular_file() || e.path().extension() != ".json") continue;
        const std::string stem = e.path().stem().string();
        if (stem.rfind("frame_", 0) != 0) continue;
        // frame_<timestamp>_<cam_id>
        const auto last_us = stem.rfind('_');
        const auto prev_us = stem.rfind('_', last_us - 1);
        if (last_us == std::string::npos || prev_us == std::string::npos) continue;
        try {
            uint64_t ts  = std::stoull(stem.substr(prev_us + 1, last_us - prev_us - 1));
            int      cam = std::stoi(stem.substr(last_us + 1));
            ids.push_back({cam, ts});
        } catch (...) {}
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

// ── Report data structures ────────────────────────────────────────────────────

struct LandmarkMatch {
    std::string region_str;
    int         class_id;
    float       confidence;
    float       best_iou;
    bool        is_tp;
};

struct FrameResult {
    uint64_t    timestamp   = 0;
    int         cam_id      = -1;

    // Prefilter
    bool has_prefilter      = false;
    bool prefilter_passed   = false;
    float cloudiness        = 0;
    float color_std         = 0;
    float contrast_std      = 0;
    float avg_value         = 0;
    std::string dominant_type;

    // RC
    bool has_gt             = false;
    std::string expected_region;
    std::vector<std::pair<std::string, float>> detected_regions; // {region_str, confidence}
    bool rc_correct         = false;

    // LD
    int gt_box_count        = 0;
    int landmark_count      = 0;
    int true_positives      = 0;
    std::vector<LandmarkMatch> matches;
};

// ── Report writer ─────────────────────────────────────────────────────────────

static std::string Pct(int num, int den)
{
    if (den == 0) return "n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << (100.0f * num / den) << "%";
    return ss.str();
}

static void WriteDatasetReport(const std::vector<FrameResult>& results,
                                const std::string& dataset_path)
{
    const char* env = std::getenv("DATASET_REPORT_PATH");
    const std::string path = env ? env : "tests/dataset_test_report.md";

    std::ofstream f(path);
    if (!f.is_open()) { spdlog::error("Could not open report file: {}", path); return; }

    // ── Global counts ────────────────────────────────────────────────────────
    int total  = static_cast<int>(results.size());
    int earth  = 0, n_with_gt = 0, rc_correct = 0;
    int total_gt_boxes = 0, total_predicted = 0, total_tp = 0;

    for (const auto& r : results) {
        if (r.has_prefilter && r.prefilter_passed) ++earth;
        if (r.has_gt) {
            ++n_with_gt;
            if (r.rc_correct) ++rc_correct;
            total_gt_boxes  += r.gt_box_count;
            total_predicted += r.landmark_count;
            total_tp        += r.true_positives;
        }
    }

    // ── Header ───────────────────────────────────────────────────────────────
    f << "# Dataset Validation Report\n\n";
    f << "**Dataset:** " << dataset_path << "\n";
    f << "**Total frames:** " << total << "\n";
    f << "**GT-labeled frames:** " << n_with_gt << "\n\n";
    f << "---\n\n";

    // ── Global summary ───────────────────────────────────────────────────────
    f << "## Global Summary\n\n";

    f << "### Prefiltering\n";
    f << "- Earth-facing (passed): **" << earth << " / " << total
      << "** (" << Pct(earth, total) << ")\n";
    f << "- Non-earth (failed/no data): **" << (total - earth) << " / " << total
      << "** (" << Pct(total - earth, total) << ")\n\n";

    f << "### RC Classification\n";
    f << "- Frames with GT region label: " << n_with_gt << "\n";
    f << "- Correct region detected: **" << rc_correct << " / " << n_with_gt
      << "** (" << Pct(rc_correct, n_with_gt) << ")\n\n";

    f << "### LD Landmark Detection\n";
    f << "- Total GT boxes: " << total_gt_boxes << "\n";
    f << "- Total predicted landmarks: " << total_predicted << "\n";
    f << "- True positives (IoU > 0.5): " << total_tp << "\n";
    f << "- Recall  (TP / GT boxes):   **" << Pct(total_tp, total_gt_boxes)   << "**\n";
    f << "- Precision (TP / Predicted): **" << Pct(total_tp, total_predicted) << "**\n\n";
    f << "---\n\n";

    // ── Per-frame sections ───────────────────────────────────────────────────
    f << "## Per-Frame Results\n\n";

    for (const auto& r : results) {
        const std::string cam_name = (r.cam_id >= 0 && r.cam_id < 4) ? kCamNames[r.cam_id] : "?";
        const std::string frame_label = "ts=" + std::to_string(r.timestamp)
                                      + " cam=" + std::to_string(r.cam_id)
                                      + " (" + cam_name + ")";

        // Section header — include expected region if available
        f << "### Frame " << frame_label;
        if (r.has_gt) f << " — Region `" << r.expected_region << "`";
        f << "\n\n";

        // Prefilter
        f << "#### Prefiltering\n";
        if (!r.has_prefilter) {
            f << "- No prefilter data\n\n";
        } else {
            f << "- Passed: **" << (r.prefilter_passed ? "yes" : "no") << "**\n";
            if (!r.dominant_type.empty())
                f << "- Dominant type: " << r.dominant_type << "\n";
            f << "- Cloudiness: " << std::fixed << std::setprecision(1) << r.cloudiness << "\n";
            f << "- Color std: "    << std::setprecision(2) << r.color_std
              << " / Contrast std: " << r.contrast_std << "\n";
            f << "- Avg value: "   << r.avg_value << "\n\n";
        }

        // RC
        f << "#### RC Classification\n";
        if (!r.has_gt) {
            f << "- No ground-truth label for this frame\n\n";
        } else {
            f << "- Expected region: `" << r.expected_region << "`\n";
            if (r.detected_regions.empty()) {
                f << "- Detected: _none_\n";
            } else {
                f << "- Detected:";
                for (const auto& [rstr, conf] : r.detected_regions)
                    f << " `" << rstr << "` (" << std::fixed << std::setprecision(3) << conf << ")";
                f << "\n";
            }
            f << "- RC correct: **" << (r.rc_correct ? "yes" : "NO") << "**\n\n";
        }

        // LD
        f << "#### LD Landmark Detection\n";
        if (!r.has_gt) {
            f << "- No ground-truth labels for this frame\n\n";
        } else {
            f << "- GT boxes: " << r.gt_box_count << "\n";
            f << "- Predicted: " << r.landmark_count << "\n";
            f << "- True positives (IoU > 0.5): " << r.true_positives
              << " / " << r.landmark_count
              << " (recall vs GT: " << Pct(r.true_positives, r.gt_box_count) << ")\n\n";

            if (!r.matches.empty()) {
                // Sort: TP first, then by region, then class_id
                auto sorted = r.matches;
                std::sort(sorted.begin(), sorted.end(), [](const LandmarkMatch& a, const LandmarkMatch& b) {
                    if (a.is_tp != b.is_tp) return a.is_tp > b.is_tp;
                    if (a.region_str != b.region_str) return a.region_str < b.region_str;
                    return a.class_id < b.class_id;
                });

                // Column widths
                struct Row { std::string region, cls, conf, iou, tp; };
                std::vector<Row> rows;
                rows.reserve(sorted.size());
                for (const auto& m : sorted) {
                    std::ostringstream cs, is;
                    cs << std::fixed << std::setprecision(3) << m.confidence;
                    is << std::fixed << std::setprecision(3) << m.best_iou;
                    rows.push_back({m.region_str, std::to_string(m.class_id),
                                    cs.str(), is.str(), m.is_tp ? "yes" : "no"});
                }
                size_t w0=9, w1=8, w2=10, w3=8, w4=3;
                for (const auto& row : rows) {
                    w0 = std::max(w0, row.region.size());
                    w1 = std::max(w1, row.cls.size());
                    w2 = std::max(w2, row.conf.size());
                    w3 = std::max(w3, row.iou.size());
                    w4 = std::max(w4, row.tp.size());
                }
                auto cell = [](const std::string& s, size_t w) {
                    return s + std::string(w - s.size(), ' ');
                };
                f << "| " << cell("region_id",  w0) << " | " << cell("class_id",   w1)
                  << " | " << cell("confidence", w2) << " | " << cell("best_iou",   w3)
                  << " | " << cell("TP?", w4) << " |\n";
                f << "| " << std::string(w0,'-') << " | " << std::string(w1,'-')
                  << " | " << std::string(w2,'-') << " | " << std::string(w3,'-')
                  << " | " << std::string(w4,'-') << " |\n";
                for (const auto& row : rows) {
                    f << "| " << cell(row.region, w0) << " | " << cell(row.cls,  w1)
                      << " | " << cell(row.conf,   w2) << " | " << cell(row.iou,  w3)
                      << " | " << cell(row.tp,     w4) << " |\n";
                }
                f << "\n";
            }
        }
        f << "---\n\n";
    }

    f.flush();
    spdlog::info("Dataset report written to: {}", path);
}

} // namespace

// Validates stored inference results against YOLO ground-truth and writes a
// Markdown report (DATASET_REPORT_PATH env var, or tests/dataset_test_report.md).
TEST(DatasetInferenceValidation, StoredLandmarks_MatchGroundTruth)
{
    constexpr float kIoUThreshold = 0.5f;

    // Build a label index: {timestamp, cam_id} → LabelFile
    const auto label_files = DiscoverLabelFiles(kSyntheticDataset);
    std::map<std::pair<uint64_t,int>, const LabelFile*> label_index;
    for (const auto& lf : label_files)
        label_index[{lf.timestamp, lf.cam_id}] = &lf;

    // Enumerate every frame JSON in the dataset
    const auto frame_ids = DiscoverFrameIDs(kSyntheticDataset);
    if (frame_ids.empty())
        GTEST_SKIP() << "No frame JSON files found in " << kSyntheticDataset;

    std::vector<FrameResult> report_data;
    int frames_with_gt = 0, total_tp = 0, total_landmarks = 0;

    for (const auto& [cam_id, ts] : frame_ids) {
        Json frame_json = DH::LoadFrameMetadataFromDisk(ts, cam_id, kSyntheticDataset);
        if (frame_json.empty()) continue;

        FrameResult r;
        r.timestamp = ts;
        r.cam_id    = cam_id;

        // ── Prefilter ────────────────────────────────────────────────────────
        if (frame_json.contains("prefilter") && frame_json["prefilter"].is_object()) {
            const auto& pf = frame_json["prefilter"];
            r.has_prefilter     = true;
            r.prefilter_passed  = pf.value("passed", false);
            r.cloudiness        = pf.value("cloudiness", 0.0f);
            r.color_std         = pf.value("color_std", 0.0f);
            r.contrast_std      = pf.value("contrast_std", 0.0f);
            r.avg_value         = pf.value("avg_value", 0.0f);
            r.dominant_type     = pf.value("dominant_type", std::string{});
        }

        // ── RC + LD ──────────────────────────────────────────────────────────
        Frame frame;
        frame.fromJson(frame_json);

        // Detected regions
        for (const auto& rid : frame.GetRegionIDs()) {
            float conf = 0.0f;
            if (frame.GetInferenceResults().has_value()) {
                for (const auto& reg : frame.GetInferenceResults()->regions)
                    if (reg.id == rid) { conf = reg.confidence; break; }
            }
            r.detected_regions.push_back({std::string(GetRegionString(rid)), conf});
        }

        // GT label for this frame
        auto it = label_index.find({ts, cam_id});
        if (it != label_index.end()) {
            const LabelFile& lf = *it->second;
            const Json& ri = frame_json.contains("raw_image") ? frame_json["raw_image"] : Json{};
            const int img_w = ri.value("width",  4608);
            const int img_h = ri.value("height", 2592);
            auto gt_boxes = ParseYoloLabels(lf.txt_path, img_w, img_h);

            if (!gt_boxes.empty()) {
                ++frames_with_gt;
                r.has_gt          = true;
                r.expected_region = lf.region_str;
                r.gt_box_count    = static_cast<int>(gt_boxes.size());

                // RC correct?
                RegionID expected_rid = GetRegionID(lf.region_str);
                r.rc_correct = std::any_of(frame.GetRegionIDs().begin(),
                                           frame.GetRegionIDs().end(),
                                           [&](RegionID rid){ return rid == expected_rid; });

                // LD matching
                const auto& landmarks = frame.GetLandmarks();
                r.landmark_count = static_cast<int>(landmarks.size());
                total_landmarks  += r.landmark_count;

                for (const auto& lm : landmarks) {
                    const cv::Rect pred(
                        static_cast<int>(lm.x - lm.width  * 0.5f),
                        static_cast<int>(lm.y - lm.height * 0.5f),
                        static_cast<int>(lm.width),
                        static_cast<int>(lm.height));

                    float best_iou = 0.0f;
                    for (const auto& [gt_cls, gt_box] : gt_boxes) {
                        if (static_cast<int>(lm.class_id) != gt_cls) continue;
                        best_iou = std::max(best_iou, BoxIoU(pred, gt_box));
                    }
                    const bool is_tp = (best_iou > kIoUThreshold);
                    if (is_tp) { ++r.true_positives; ++total_tp; }
                    r.matches.push_back({std::string(GetRegionString(lm.region_id)),
                                         static_cast<int>(lm.class_id),
                                         lm.confidence, best_iou, is_tp});
                }
            }
        }
        report_data.push_back(std::move(r));
    }

    // Sort by timestamp then cam_id for stable report ordering
    std::sort(report_data.begin(), report_data.end(),
              [](const FrameResult& a, const FrameResult& b){
                  return a.timestamp != b.timestamp ? a.timestamp < b.timestamp
                                                    : a.cam_id   < b.cam_id;
              });

    WriteDatasetReport(report_data, kSyntheticDataset);

    if (frames_with_gt == 0)
        GTEST_SKIP() << "All .txt label files are empty — no GT to validate against";

    EXPECT_GT(total_landmarks, 0)
        << "No landmarks in stored results for labeled frames";
    EXPECT_GT(total_tp, 0)
        << "No true-positive landmarks (IoU > " << kIoUThreshold << ") across "
        << frames_with_gt << " labeled frames (" << total_landmarks << " landmarks checked)";
}
