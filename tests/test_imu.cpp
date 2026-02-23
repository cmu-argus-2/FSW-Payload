#include <gtest/gtest.h>
#include <imu/imu_manager.hpp>

// Test Data Collection configuration
TEST(IMUManagerTest, DataCollectionConfig) 
{
    IMUManager imuManager;

    // Test setters
    imuManager.SetSampleRate(25.0f);
    EXPECT_EQ(imuManager.GetSampleRate(), 25.0f);
    imuManager.SetLogFile("test_imu_log.txt");
    EXPECT_EQ(imuManager.GetLogFile(), "test_imu_log.txt");

    imuManager.StartCollection();
    ASSERT_EQ(imuManager.GetIMUManagerStatus(), static_cast<uint8_t>(IMU_STATE::COLLECT));
    
    // Since we don't have direct access to the BMI160 class from the test, we can check the sensor status to infer power modes
    uint8_t pmu_status;
    int32_t statusRet = imuManager.ReadPowerModeStatus(&pmu_status);
    EXPECT_EQ(statusRet, BMI160::RTN_NO_ERROR);

    // Assert that the gyro and magnetometer are in the correct power mode
    BMI160::PowerModes magStatus, gyroStatus, accStatus;
    BMI160::decodePowerModeStatus(pmu_status, &magStatus, &gyroStatus, &accStatus);
    EXPECT_EQ(magStatus, BMI160::NORMAL);
    EXPECT_EQ(gyroStatus, BMI160::NORMAL);
    EXPECT_EQ(accStatus, BMI160::SUSPEND);

    // Assert that the magnetometer is in normal mode and providing data
    BMI160::SensorData magData;
    int32_t magRet = imuManager.ReadMagnetometerData(magData);
    spdlog::info("Magnetometer read return code in suspend mode: {}, {}, {}, {}", 
        magRet, magData.xAxis.scaled, magData.yAxis.scaled, magData.zAxis.scaled);
    EXPECT_EQ(magRet, BMI160::RTN_NO_ERROR);

    // Assert that the gyro is in normal mode and providing data
    BMI160::SensorData gyroData;
    int32_t gyroRet = imuManager.ReadGyroData(gyroData);
    spdlog::info("Gyroscope read return code in suspend mode: {}, {}, {}, {}", 
        gyroRet, gyroData.xAxis.scaled, gyroData.yAxis.scaled, gyroData.zAxis.scaled);
    EXPECT_EQ(gyroRet, BMI160::RTN_NO_ERROR);

    // Assert that we can still read temperature data since gyro is in fast start-up mode
    float temperature;
    int32_t tempRet = imuManager.ReadTemperatureData(&temperature);
    spdlog::info("Temperature read return code in suspend mode: {}, temperature: {}", tempRet, temperature);
    EXPECT_EQ(tempRet, BMI160::RTN_NO_ERROR);

    // TODO: Assert that the gyro is configured correctly (range, bdw, ODR)


}

// Test Suspend Mode Setup
TEST(IMUManagerTest, SuspendModeSetup) 
{
    // Its initialized in IDLE mode. Suspend method runs on initialization
    IMUManager imuManager;
    EXPECT_EQ(imuManager.GetIMUManagerStatus(), static_cast<uint8_t>(IMU_STATE::IDLE));

    // Test that the power mode status function works while idle
    uint8_t pmu_status;
    int32_t statusRet = imuManager.ReadPowerModeStatus(&pmu_status);
    EXPECT_EQ(statusRet, BMI160::RTN_NO_ERROR);

    // Test that the sensors are in the correct power mode on idle
    BMI160::PowerModes magStatus, gyroStatus, accStatus;
    BMI160::decodePowerModeStatus(pmu_status, &magStatus, &gyroStatus, &accStatus);
    EXPECT_EQ(magStatus, BMI160::SUSPEND);
    EXPECT_EQ(gyroStatus, BMI160::FAST_START_UP);
    EXPECT_EQ(accStatus, BMI160::SUSPEND);

    // Assert that we can still read temperature data since gyro is in fast start-up mode
    float temperature;
    int32_t tempRet = imuManager.ReadTemperatureData(&temperature);
    spdlog::info("Temperature read return code in suspend mode: {}, temperature: {}", tempRet, temperature);
    EXPECT_EQ(tempRet, BMI160::RTN_NO_ERROR);
}

// Test Data Collection and Logging
TEST(IMUManagerTest, DataCollectionAndLogging) 
{
    IMUManager imuManager;
    std::thread imuThread(&IMUManager::RunLoop, &imuManager);

    imuManager.SetSampleRate(25.0f);
    imuManager.SetLogFile("test_imu_log.txt");
    imuManager.StartCollection();
    
    // Let it collect for a short duration
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    imuManager.Suspend();
    imuManager.StopLoop();
    if (imuThread.joinable()) {
        imuThread.join();
    }
    
    // Check that the log file was created and has content
    std::ifstream logFile("test_imu_log.txt");
    EXPECT_TRUE(logFile.is_open());
    std::string line;
    int lineCount = 0;
    // TODO: Implement function to parse imu log file for a dataset
    while (std::getline(logFile, line)) {
        lineCount++;
        // timestamp, gyro x, gyro y, gyro z, mag x, mag y, mag z, temperature
        std::istringstream iss(line);
        uint64_t timestamp;
        float gyroX, gyroY, gyroZ, magX, magY, magZ, temperature;
        char comma; // to consume the commas
        if (!(iss >> timestamp >> comma >> gyroX >> comma >> gyroY >> comma >> gyroZ >> comma >> magX >> comma >> magY >> comma >> magZ >> comma >> temperature)) {
            // If parsing fails, the line format is incorrect
            FAIL() << "Log line format incorrect: " << line;
        }
    }
    EXPECT_GT(lineCount, 24); // Expect at least one data line (header + data)

    logFile.close();
    // std::remove("test_imu_log.txt");
}

// Test Suspend Mode Telemetry data retrieval
TEST(IMUManagerTest, SuspendModeTelemetryDataRetrieval) 
{
    IMUManager imuManager;
    imuManager.Suspend();
    // get temperature and imu manager status
    EXPECT_TRUE(true); // Placeholder assertion to ensure test runs
}

// TODO: Error handling tests