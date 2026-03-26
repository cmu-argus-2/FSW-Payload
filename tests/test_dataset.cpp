/*
    Test file to replicate what should happen in FSW when the command to start 
    dataset collection is received.
*/
#include "spdlog/spdlog.h"
#include "vision/dataset_manager.hpp"
#include "core/timing.hpp"
#include "configuration.hpp"
#include <memory>
#include <thread>
#include <array>

#include <gtest/gtest.h>
#include <core/errors.hpp>

#define DATASET_KEY_CMD "CMD"

#define ERASE_TEST_FILES false

namespace fs = std::filesystem;


// Test to check that the dataset configuration parameters are correctly validated 
// and that invalid configurations are rejected
TEST(DatasetTest, DatasetConfigurationCheck) 
{
    // configuration parameters
    double max_period = 60.0;
    uint16_t nb_frames = 100;
    CAPTURE_MODE capture_mode = CAPTURE_MODE::PERIODIC;
    IMU_COLLECTION_MODE imu_collection_mode = IMU_COLLECTION_MODE::GYRO_ONLY;
    uint8_t image_capture_rate = 60;
    float imu_sample_rate_hz = 1.0f;
    ProcessingStage target_processing_stage = ProcessingStage::NotPrefiltered;
    uint64_t capture_start_time = timing::GetCurrentTimeMs();

    EXPECT_TRUE(Dataset::isValidConfiguration(max_period, nb_frames, capture_mode, imu_collection_mode, image_capture_rate, 
                                                imu_sample_rate_hz, target_processing_stage, capture_start_time));
    
    max_period = 0.0;
    EXPECT_FALSE(Dataset::isValidConfiguration(max_period, nb_frames, capture_mode, imu_collection_mode, image_capture_rate, 
                                                imu_sample_rate_hz, target_processing_stage, capture_start_time));
    
    max_period = 60.0; nb_frames  = 0;
    EXPECT_FALSE(Dataset::isValidConfiguration(max_period, nb_frames, capture_mode, imu_collection_mode, image_capture_rate, 
                                                imu_sample_rate_hz, target_processing_stage, capture_start_time));
    
    nb_frames = 11000;                    
    EXPECT_FALSE(Dataset::isValidConfiguration(max_period, nb_frames, capture_mode, imu_collection_mode, image_capture_rate, 
                                                imu_sample_rate_hz, target_processing_stage, capture_start_time));

    // should it be possible to collect a dataset with no images? could still capture imu data
    nb_frames = 100; capture_mode = CAPTURE_MODE::IDLE; 
    EXPECT_FALSE(Dataset::isValidConfiguration(max_period, nb_frames, capture_mode, imu_collection_mode, image_capture_rate, 
                                                imu_sample_rate_hz, target_processing_stage, capture_start_time));
    
    capture_mode = CAPTURE_MODE::CAPTURE_SINGLE;
    EXPECT_FALSE(Dataset::isValidConfiguration(max_period, nb_frames, capture_mode, imu_collection_mode, image_capture_rate, 
                                                imu_sample_rate_hz, target_processing_stage, capture_start_time));

    capture_mode = CAPTURE_MODE::PERIODIC;
    EXPECT_TRUE(Dataset::isValidConfiguration(max_period, nb_frames, capture_mode, imu_collection_mode, image_capture_rate, 
                                                imu_sample_rate_hz, target_processing_stage, capture_start_time));
    
    image_capture_rate = 0;
    EXPECT_FALSE(Dataset::isValidConfiguration(max_period, nb_frames, capture_mode, imu_collection_mode, image_capture_rate, 
                                                imu_sample_rate_hz, target_processing_stage, capture_start_time));
    
    imu_sample_rate_hz = 0.0f; image_capture_rate = 60;
    EXPECT_FALSE(Dataset::isValidConfiguration(max_period, nb_frames, capture_mode, imu_collection_mode, image_capture_rate, 
                                                imu_sample_rate_hz, target_processing_stage, capture_start_time));

    // should there be error handling for capture_start_time?
    
    // Test input based constructor
    // TODO: Check with invalid configuration that an std::invalid_argument exception is thrown
    spdlog::info("before assert no throw");
    imu_sample_rate_hz = 1.0f;
    ASSERT_NO_THROW(Dataset(max_period, nb_frames, capture_mode, 
                            imu_collection_mode, image_capture_rate, 
                            imu_sample_rate_hz, target_processing_stage, 
                            capture_start_time));
    spdlog::info("after assert no throw, before redeclaration");
    // TODO: What if a dataset is created like this, and the folder and config already exist?
    // Difference between blocking the definition of a new dataset that intersects another 
    // and allowing the redefinition of an instance from a file. 
    Dataset dataset(max_period, nb_frames, capture_mode, 
                    imu_collection_mode, image_capture_rate, 
                    imu_sample_rate_hz, target_processing_stage, 
                    capture_start_time);

    // Test getters
    EXPECT_EQ(dataset.GetCaptureStartTime(), capture_start_time);
    EXPECT_EQ(dataset.GetTargetFrameNb(), nb_frames);
    EXPECT_EQ(dataset.GetMaximumPeriod(), max_period);
    EXPECT_EQ(dataset.GetDatasetCaptureMode(), capture_mode);
    EXPECT_EQ(dataset.GetIMUCollectionMode(), imu_collection_mode);
    EXPECT_EQ(dataset.GetImageCaptureRate(), image_capture_rate);
    EXPECT_EQ(dataset.GetIMUSampleRateHz(), imu_sample_rate_hz);
    EXPECT_EQ(dataset.GetTargetProcessingStage(), target_processing_stage);

    std::string folder_path   = "data/datasets/" + std::to_string(capture_start_time) + "/";
    std::string imu_file_path = folder_path + "imu_data.csv";

    EXPECT_EQ(dataset.GetFolderPath(), folder_path);
    EXPECT_EQ(dataset.GetIMUFilePath(), imu_file_path);
    // TODO: calling GetStoredFrameIDs when stored_frame_ids hasn't been initialized results in segmentation fault.
    // Need to handle that
    // EXPECT_TRUE(dataset.GetStoredFrameIDs().empty());
    

    // Test CreateConfigurationFile (ran from constructor if constructor detects valid config)
    std::string dataset_config_file_path = folder_path  + "dataset_config.toml";
    EXPECT_TRUE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    // Check that below runs without throwing errors
    ASSERT_NO_THROW(Dataset{folder_path});

    toml::table config = toml::parse_file(dataset_config_file_path);

    config.erase("maximum_period");
    // save file again
    std::ofstream file(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);
    
    config.insert("maximum_period", 0.0f);
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);

    config.erase("maximum_period"); config.insert("maximum_period", 60.0f);
    config.erase("target_frames_nb"); 
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);
    
    config.insert("target_frames_nb", 0);
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);

    config.erase("target_frames_nb"); config.insert("target_frames_nb", 100);
    config.erase("dataset_capture_mode"); 
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);
    
    config.insert("dataset_capture_mode", 0);
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);

    config.erase("dataset_capture_mode"); config.insert("dataset_capture_mode", 2);
    config.erase("imu_collection_mode"); 
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);
    
    config.insert("imu_collection_mode",-1);
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);

    config.erase("imu_collection_mode"); config.insert("imu_collection_mode", 2);
    config.erase("image_capture_rate");
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);
    
    config.insert("image_capture_rate",-1);
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);

    config.erase("image_capture_rate"); config.insert("image_capture_rate", 60);
    config.erase("imu_sample_rate_hz");
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);
    
    config.insert("imu_sample_rate_hz",-1);
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);

    config.erase("imu_sample_rate_hz"); config.insert("imu_sample_rate_hz", 1.0f);
    config.erase("target_processing_stage");
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);
    
    config.insert("target_processing_stage",-1);
    file.open(dataset_config_file_path, std::ofstream::out | std::ofstream::trunc);
    file << config;
    file.close();
    EXPECT_FALSE(Dataset::isValidConfigurationFile(dataset_config_file_path));
    ASSERT_THROW(Dataset{folder_path}, std::invalid_argument);

    // Option to delete generated test data
    if (ERASE_TEST_FILES) {
        try {
            // remove_all recursively deletes all contents and the folder itself
            unsigned long long count = fs::remove_all(folder_path); 
            std::cout << "Successfully deleted " << count << " items in " << folder_path << std::endl;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error deleting folder: " << e.what() << std::endl;
        }
    }
    
}

/**/

TEST(DatasetTest, DatasetStorageAndRetrieval) 
{
    // First make the dataset

    // configuration parameters
    double max_period = 60.0;
    uint16_t nb_frames = 100;
    CAPTURE_MODE capture_mode = CAPTURE_MODE::PERIODIC;
    IMU_COLLECTION_MODE imu_collection_mode = IMU_COLLECTION_MODE::GYRO_ONLY;
    uint8_t image_capture_rate = 60;
    float imu_sample_rate_hz = 1.0f;
    ProcessingStage target_processing_stage = ProcessingStage::NotPrefiltered;
    uint64_t capture_start_time = timing::GetCurrentTimeMs();

    Dataset dataset(max_period, nb_frames, capture_mode, 
                    imu_collection_mode, image_capture_rate, 
                    imu_sample_rate_hz, target_processing_stage, 
                    capture_start_time);

    // Test that dataset is stored in the right place
    auto folder_path = "data/datasets/" + std::to_string(capture_start_time);
    EXPECT_TRUE(fs::exists(folder_path));
    EXPECT_TRUE(fs::exists(folder_path + "/dataset_config.toml"));

    // Test that dataset can be retrieved from folder path and that parameters are correctly read
    Dataset retrieved_dataset(folder_path);
    EXPECT_EQ(retrieved_dataset.GetMaximumPeriod(), 60.0);
    EXPECT_EQ(retrieved_dataset.GetTargetFrameNb(), 100);
    EXPECT_EQ(retrieved_dataset.GetDatasetCaptureMode(), CAPTURE_MODE::PERIODIC);
    EXPECT_EQ(retrieved_dataset.GetCaptureStartTime(), capture_start_time);
    EXPECT_EQ(retrieved_dataset.GetIMUCollectionMode(), IMU_COLLECTION_MODE::GYRO_ONLY);
    EXPECT_EQ(retrieved_dataset.GetImageCaptureRate(), 60);
    EXPECT_EQ(retrieved_dataset.GetIMUSampleRateHz(), 1.0f);
    EXPECT_EQ(retrieved_dataset.GetTargetProcessingStage(), ProcessingStage::NotPrefiltered);

    // Error handling: dataset doesn't exist
    ASSERT_THROW(Dataset{"data/datasets/bogus_dataset_path/"}, std::invalid_argument);

    // Test that dataset can be converted to and from json correctly
    Json j = dataset.toJson();
    EXPECT_TRUE(j.contains("folder_path"));
    EXPECT_TRUE(j.contains("capture_start_time"));
    EXPECT_TRUE(j.contains("maximum_period"));
    EXPECT_TRUE(j.contains("target_frame_nb"));
    EXPECT_TRUE(j.contains("dataset_capture_mode"));
    EXPECT_TRUE(j.contains("imu_collection_mode"));
    EXPECT_TRUE(j.contains("image_capture_rate"));
    EXPECT_TRUE(j.contains("imu_sample_rate_hz"));
    EXPECT_TRUE(j.contains("target_processing_stage"));
    EXPECT_TRUE(j.contains("imu_log_file_path"));
    EXPECT_TRUE(j.contains("frame_id_list"));

    EXPECT_EQ(j["capture_start_time"], capture_start_time);
    EXPECT_EQ(j["maximum_period"], 60.0);
    EXPECT_EQ(j["target_frame_nb"], 100);

    Dataset dataset_from_json(max_period, nb_frames, capture_mode,
                          imu_collection_mode, image_capture_rate,
                          imu_sample_rate_hz, target_processing_stage,
                          capture_start_time + 100000);
    EXPECT_TRUE(dataset_from_json.fromJson(j));
    EXPECT_EQ(dataset_from_json.GetCaptureStartTime(), capture_start_time);
    EXPECT_EQ(dataset_from_json.GetMaximumPeriod(), 60.0);
    EXPECT_EQ(dataset_from_json.GetTargetFrameNb(), 100);
    EXPECT_EQ(dataset_from_json.GetFolderPath(), folder_path + "/");
}


// Test to check that the dataset manager configuration parameters are correctly validated 
// and that invalid configurations are rejected
TEST(DatasetTest, DatasetManagerConfigurationCheck) 
{
    // Test error handling of definition of two overlapping datasets
    // Test configuration of two datasets sequentially not overlapping, and that active datasets outputs the right values
    // Optional: Test that a dataset can handle its images/data no longer being available
}

TEST(DatasetTest, DatasetReprocessing) 
{
    // Test reprocessing a dataset
}
