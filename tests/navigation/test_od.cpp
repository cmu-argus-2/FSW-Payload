// Orbit Determination Integration Tests

#include <gtest/gtest.h>
#include "spdlog/spdlog.h"
#include "navigation/od.hpp"
#include <filesystem>


// Load a dataset

// Run the dataset prepare to generate a landmark_measurements.csv file

// Run the OD batch optimization on the dataset and verify that it completes without error

// TODO: Change the OD interface to use the landmark measurements from the CSV. Remove HDF5 dependency from pre-processing in the code

TEST_F(OrchestratorTest, SetLDNetConfig_Reinit_NoGPUMemoryLeak)
{

}