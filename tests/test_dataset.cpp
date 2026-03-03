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

#define DATASET_KEY_CMD "CMD"

#include <gtest/gtest.h>
#include <core/errors.hpp>

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
    imu_sample_rate_hz = 1.0f; 
    Dataset dataset(max_period, nb_frames, capture_mode, imu_collection_mode, image_capture_rate, 
                                                imu_sample_rate_hz, target_processing_stage, capture_start_time);

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
    

    // Test CreateConfigurationFile (ran from constructor)
    EXPECT_TRUE(Dataset::isValidConfigurationFile(folder_path  + "dataset_config.toml"));

    // TODO: change config.toml to check error handling


    // Test isValidConfigurationFile from generated toml files
    // std::string valid_config_path = "tests/test_data/valid_dataset_config.toml";
    // EXPECT_TRUE(isValidConfigurationFile(valid_config_path));
    // std::string invalid_config_path = "tests/test_data/invalid_dataset_config.toml";
    // EXPECT_FALSE(isValidConfigurationFile(invalid_config_path));

    // Test destructor

    // Test Configuration file-based constructor

    // Option to delete generated test data

}

/**/

TEST(DatasetTest, DatasetStorageAndRetrieval) 
{
    // Test that dataset is stored in the right place

    // Test that dataset can be retrieved from folder path and that parameters are correctly read

    // Error handling: dataset doesn't exist

    // Test that dataset can be converted to and from json correctly
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
