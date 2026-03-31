/*
    Test file to replicate what should happen in FSW when the command to start
    dataset collection is received.
*/
#include "spdlog/spdlog.h"
#include "vision/dataset_manager.hpp"
#include "core/timing.hpp"
#include "configuration.hpp"
#include <fstream>
#include <functional>

#include <gtest/gtest.h>
#include <core/errors.hpp>

#define ERASE_TEST_FILES false

namespace fs = std::filesystem;

// ── Fixture ───────────────────────────────────────────────────────────────────

struct DatasetTest : ::testing::Test
{
    double              max_period              = 60.0;
    uint16_t            nb_frames               = 100;
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

    // Save field, set bad value, assert invalid, restore.
    auto checkInvalid = [&](auto& field, auto bad_val) {
        auto saved = field;
        field = bad_val;
        EXPECT_FALSE(isValid());
        field = saved;
    };

    checkInvalid(max_period,         0.0);
    checkInvalid(nb_frames,          uint16_t(0));
    checkInvalid(nb_frames,          uint16_t(11000));
    checkInvalid(capture_mode,       CAPTURE_MODE::IDLE);
    checkInvalid(capture_mode,       CAPTURE_MODE::CAPTURE_SINGLE);
    checkInvalid(image_capture_rate, uint8_t(0));
    checkInvalid(imu_sample_rate_hz, 0.0f);

    // should there be error handling for capture_start_time?

    // Constructor
    // TODO: Check with invalid configuration that an std::invalid_argument exception is thrown
    ASSERT_NO_THROW(makeDataset());
    Dataset dataset = makeDataset();

    // Getters
    EXPECT_EQ(dataset.GetCaptureStartTime(),      capture_start_time);
    EXPECT_EQ(dataset.GetTargetFrameNb(),         nb_frames);
    EXPECT_EQ(dataset.GetMaximumPeriod(),         max_period);
    EXPECT_EQ(dataset.GetDatasetCaptureMode(),    capture_mode);
    EXPECT_EQ(dataset.GetIMUCollectionMode(),     imu_collection_mode);
    EXPECT_EQ(dataset.GetImageCaptureRate(),      image_capture_rate);
    EXPECT_EQ(dataset.GetIMUSampleRateHz(),       imu_sample_rate_hz);
    EXPECT_EQ(dataset.GetTargetProcessingStage(), target_processing_stage);
    // TODO: calling GetStoredFrameIDs when stored_frame_ids hasn't been initialized
    // results in segmentation fault — needs handling before this can be tested.
    // EXPECT_TRUE(dataset.GetStoredFrameIDs().empty());

    const std::string folder_path = "data/datasets/" + std::to_string(capture_start_time) + "/";
    EXPECT_EQ(dataset.GetFolderPath(),  folder_path);
    EXPECT_EQ(dataset.GetIMUFilePath(), folder_path + "imu_data.csv");

    // Config file: valid on creation, invalid after corruption
    // TODO: What if a dataset is created like this, and the folder and config already exist?
    const std::string config_path = folder_path + "dataset_config.toml";
    EXPECT_TRUE(Dataset::isValidConfigurationFile(config_path));
    ASSERT_NO_THROW(Dataset{folder_path});

    toml::table config = toml::parse_file(config_path);

    auto writeConfig = [&]() {
        std::ofstream f(config_path, std::ofstream::out | std::ofstream::trunc);
        f << config;
    };
    auto expectInvalidConfig = [&]() {
        EXPECT_FALSE(Dataset::isValidConfigurationFile(config_path));
        EXPECT_THROW(Dataset{folder_path}, std::invalid_argument);
    };

    // Table-driven: for each field verify (a) missing → invalid, (b) bad value → invalid.
    // insert_good restores the field so the next iteration starts from a valid config.
    struct FieldTest {
        std::string field;
        std::function<void(toml::table&)> insert_bad;
        std::function<void(toml::table&)> insert_good; // empty for last entry
    };

    const std::vector<FieldTest> field_tests = {
        {"maximum_period",
            [](toml::table& c){ c.insert("maximum_period",           0.0f); },
            [](toml::table& c){ c.erase("maximum_period");  c.insert("maximum_period",         60.0f); }},
        {"target_frame_nb",
            [](toml::table& c){ c.insert("target_frame_nb",          0); },
            [](toml::table& c){ c.erase("target_frame_nb"); c.insert("target_frame_nb",        100); }},
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
        config.erase(field);
        writeConfig();
        expectInvalidConfig();

        insert_bad(config);
        writeConfig();
        expectInvalidConfig();

        config.erase(field);
        insert_good(config);
    }

    if (ERASE_TEST_FILES) fs::remove_all(folder_path);
}

// ── DatasetStorageAndRetrieval ────────────────────────────────────────────────

TEST_F(DatasetTest, DatasetStorageAndRetrieval)
{
    Dataset dataset = makeDataset();

    const std::string folder_path = "data/datasets/" + std::to_string(capture_start_time);
    EXPECT_TRUE(fs::exists(folder_path));
    EXPECT_TRUE(fs::exists(folder_path + "/dataset_config.toml"));

    // Round-trip through folder path
    Dataset retrieved(folder_path);
    EXPECT_EQ(retrieved.GetMaximumPeriod(),         60.0);
    EXPECT_EQ(retrieved.GetTargetFrameNb(),         100);
    EXPECT_EQ(retrieved.GetDatasetCaptureMode(),    CAPTURE_MODE::PERIODIC);
    EXPECT_EQ(retrieved.GetCaptureStartTime(),      capture_start_time);
    EXPECT_EQ(retrieved.GetIMUCollectionMode(),     IMU_COLLECTION_MODE::GYRO_ONLY);
    EXPECT_EQ(retrieved.GetImageCaptureRate(),      60);
    EXPECT_EQ(retrieved.GetIMUSampleRateHz(),       1.0f);
    EXPECT_EQ(retrieved.GetTargetProcessingStage(), ProcessingStage::NotPrefiltered);

    ASSERT_THROW(Dataset{"data/datasets/bogus_dataset_path/"}, std::invalid_argument);

    // JSON serialisation
    Json j = dataset.toJson();

    for (const char* key : {"folder_path", "capture_start_time", "maximum_period",
                             "target_frame_nb", "dataset_capture_mode", "imu_collection_mode",
                             "image_capture_rate", "imu_sample_rate_hz", "target_processing_stage",
                             "imu_log_file_path", "frame_id_list"})
    {
        EXPECT_TRUE(j.contains(key)) << "missing JSON key: " << key;
    }

    EXPECT_EQ(j["capture_start_time"], capture_start_time);
    EXPECT_EQ(j["maximum_period"],     60.0);
    EXPECT_EQ(j["target_frame_nb"],    100);

    // fromJson round-trip
    Dataset dataset_from_json(max_period, nb_frames, capture_mode, imu_collection_mode,
                              image_capture_rate, imu_sample_rate_hz, target_processing_stage,
                              capture_start_time + 100000);
    EXPECT_TRUE(dataset_from_json.fromJson(j));
    EXPECT_EQ(dataset_from_json.GetCaptureStartTime(), capture_start_time);
    EXPECT_EQ(dataset_from_json.GetMaximumPeriod(),    60.0);
    EXPECT_EQ(dataset_from_json.GetTargetFrameNb(),    100);
    EXPECT_EQ(dataset_from_json.GetFolderPath(),       folder_path + "/");
}

TEST_F(DatasetTest, DatasetManagerConfigurationCheck)
{
    // TODO: Test error handling of definition of two overlapping datasets
    // TODO: Test configuration of two sequential non-overlapping datasets, verify active dataset outputs
    // Optional: Test that a dataset can handle its images/data no longer being available
}

TEST_F(DatasetTest, DatasetReprocessing)
{
    // TODO: Test reprocessing a dataset
}

// TODO: Integration tests