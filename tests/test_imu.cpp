#include <gtest/gtest.h>
#include <imu/imu_manager.hpp>
#include "spdlog/spdlog.h"
#include <thread>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>

// Test Data Collection configuration
TEST(IMUManagerTest, DataCollectionConfig) 
{
    IMUManager imuManager(IMUConfig{
        .chipid = 0xD8,
        .i2c_addr = 0x68,
        .i2c_path = "/dev/i2c-7"
    });

    // Test setters
    imuManager.SetSampleRate(25.0f);
    EXPECT_EQ(imuManager.GetSampleRate(), 25.0f);
    imuManager.SetLogFile("test_imu_config.csv");
    EXPECT_EQ(imuManager.GetLogFile(), "test_imu_config.csv");
    imuManager.SetCollectionMode(IMU_COLLECTION_MODE::GYRO_MAG_TEMP);
    EXPECT_EQ(imuManager.GetCollectionMode(), IMU_COLLECTION_MODE::GYRO_MAG_TEMP);

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
    spdlog::info("Magnetometer read return code while collecting: {}, {}, {}, {}",
        magRet, magData.xAxis.scaled, magData.yAxis.scaled, magData.zAxis.scaled);
    EXPECT_EQ(magRet, BMI160::RTN_NO_ERROR);

    // Assert that the gyro is in normal mode and providing data
    BMI160::SensorData gyroData;
    int32_t gyroRet = imuManager.ReadGyroData(gyroData);
    spdlog::info("Gyroscope read return code while collecting: {}, {}, {}, {}",
        gyroRet, gyroData.xAxis.scaled, gyroData.yAxis.scaled, gyroData.zAxis.scaled);
    EXPECT_EQ(gyroRet, BMI160::RTN_NO_ERROR);

    // Assert that we can read temperature data while collecting
    float temperature;
    int32_t tempRet = imuManager.ReadTemperatureData(&temperature);
    spdlog::info("Temperature read return code while collecting: {}, temperature: {}", tempRet, temperature);
    EXPECT_EQ(tempRet, BMI160::RTN_NO_ERROR);

    // TODO: Assert that the gyro is configured correctly (range, bdw, ODR)

    // Test Suspend: sensors go to low-power, state returns to IDLE
    int suspendRet = imuManager.Suspend();
    EXPECT_EQ(suspendRet, 0);
    ASSERT_EQ(imuManager.GetIMUManagerStatus(), static_cast<uint8_t>(IMU_STATE::IDLE));

    uint8_t pmu_status_suspended;
    EXPECT_EQ(imuManager.ReadPowerModeStatus(&pmu_status_suspended), BMI160::RTN_NO_ERROR);
    BMI160::PowerModes magSusp, gyroSusp, accSusp;
    BMI160::decodePowerModeStatus(pmu_status_suspended, &magSusp, &gyroSusp, &accSusp);
    EXPECT_EQ(magSusp, BMI160::SUSPEND);
    EXPECT_EQ(gyroSusp, BMI160::FAST_START_UP);
    EXPECT_EQ(accSusp, BMI160::SUSPEND);

    std::remove("test_imu_config.csv");
}

// Test Suspend Mode Setup
TEST(IMUManagerTest, SuspendModeSetup) 
{
    // Its initialized in IDLE mode. Suspend method runs on initialization
    IMUManager imuManager(IMUConfig{
        .chipid = 0xD8,
        .i2c_addr = 0x68,
        .i2c_path = "/dev/i2c-7"
    });
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
    IMUManager imuManager(IMUConfig{
        .chipid = 0xD8,
        .i2c_addr = 0x68,
        .i2c_path = "/dev/i2c-7"
    });
    std::thread imuThread(&IMUManager::RunLoop, &imuManager);

    imuManager.SetSampleRate(25.0f);
    imuManager.SetLogFile("test_imu_log.csv");
    imuManager.SetCollectionMode(IMU_COLLECTION_MODE::GYRO_MAG_TEMP);
    imuManager.StartCollection();
    
    // Let it collect for a short duration
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // TODO: changing collection mode should fail while collecting
    // TODO: Implement this in a loop that runs over all collection modes
    
    imuManager.Suspend();
    imuManager.StopLoop();
    if (imuThread.joinable()) {
        imuThread.join();
    }
    
    // Check that the log file was created and has content
    std::ifstream logFile("test_imu_log.csv");
    EXPECT_TRUE(logFile.is_open());
    std::string line;
    int lineCount = 0;
    // TODO: Implement function to parse imu log file for a dataset
    while (std::getline(logFile, line)) {
        if (lineCount == 0) {
            // Check header line
            EXPECT_EQ(line, "Timestamp_ms, Gyro_X_dps, Gyro_Y_dps, Gyro_Z_dps, Mag_X_uT, Mag_Y_uT, Mag_Z_uT, Temperature_C");
            lineCount++;
            continue;
        }
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
        // data range checks (gyro range ±125 dps, mag range ±1000 uT, temperature sensor range)
        EXPECT_LE(std::abs(gyroX), 125.0f);
        EXPECT_LE(std::abs(gyroY), 125.0f);
        EXPECT_LE(std::abs(gyroZ), 125.0f);
        EXPECT_LE(std::abs(magX), 1000.0f);
        EXPECT_LE(std::abs(magY), 1000.0f);
        EXPECT_LE(std::abs(magZ), 1000.0f);
        EXPECT_GE(temperature, -40.0f); // Sensor won't operate below -40C
        EXPECT_LE(temperature, 85.0f); // Sensor won't operate above 85C
    }
    EXPECT_GT(lineCount, 24); // Expect at least one data line (header + data)

    logFile.close();
    std::remove("test_imu_log.csv");
}

// Test Suspend Mode Telemetry data retrieval
TEST(IMUManagerTest, SuspendModeTelemetryDataRetrieval) 
{
    IMUManager imuManager(IMUConfig{
        .chipid = 0xD8,
        .i2c_addr = 0x68,
        .i2c_path = "/dev/i2c-7"
    });
    imuManager.Suspend();
    // get temperature and imu manager status
    EXPECT_TRUE(true); // Placeholder assertion to ensure test runs
}

// Test that Suspend() called from main thread while RunLoop is writing does not corrupt the log file
TEST(IMUManagerTest, ThreadSafeSuspend)
{
    IMUManager imuManager(IMUConfig{
        .chipid = 0xD8,
        .i2c_addr = 0x68,
        .i2c_path = "/dev/i2c-7"
    });
    std::thread imuThread(&IMUManager::RunLoop, &imuManager);

    imuManager.SetSampleRate(25.0f);
    imuManager.SetLogFile("test_imu_thread_safety.csv");
    imuManager.SetCollectionMode(IMU_COLLECTION_MODE::GYRO_MAG_TEMP);
    imuManager.StartCollection();

    // Let RunLoop write several samples, then suspend from main thread while it is still running
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    int suspendRet = imuManager.Suspend();
    EXPECT_EQ(suspendRet, 0);
    EXPECT_EQ(imuManager.GetIMUManagerStatus(), static_cast<uint8_t>(IMU_STATE::IDLE));

    imuManager.StopLoop();
    if (imuThread.joinable()) {
        imuThread.join();
    }

    // File must be openable and start with a valid header — corruption would break this
    std::ifstream logFile("test_imu_thread_safety.csv");
    ASSERT_TRUE(logFile.is_open());
    std::string header;
    ASSERT_TRUE(std::getline(logFile, header));
    EXPECT_EQ(header, "Timestamp_ms, Gyro_X_dps, Gyro_Y_dps, Gyro_Z_dps, Mag_X_uT, Mag_Y_uT, Mag_Z_uT, Temperature_C");

    // Every data line after the header must parse cleanly — a torn write would fail here
    std::string line;
    while (std::getline(logFile, line)) {
        std::istringstream iss(line);
        uint64_t timestamp;
        float gyroX, gyroY, gyroZ, magX, magY, magZ, temperature;
        char comma;
        EXPECT_TRUE(iss >> timestamp >> comma >> gyroX >> comma >> gyroY >> comma >> gyroZ
                        >> comma >> magX >> comma >> magY >> comma >> magZ >> comma >> temperature)
            << "Torn or corrupt log line: " << line;
    }

    logFile.close();
    std::remove("test_imu_thread_safety.csv");
}

// Test that rapid Start/Suspend cycles from main thread while RunLoop runs do not crash or corrupt
TEST(IMUManagerTest, ThreadSafeStartStopCycles)
{
    IMUManager imuManager(IMUConfig{
        .chipid = 0xD8,
        .i2c_addr = 0x68,
        .i2c_path = "/dev/i2c-7"
    });
    std::thread imuThread(&IMUManager::RunLoop, &imuManager);

    imuManager.SetSampleRate(25.0f);
    imuManager.SetLogFile("test_imu_cycles.csv");
    imuManager.SetCollectionMode(IMU_COLLECTION_MODE::GYRO_MAG_TEMP);

    for (int i = 0; i < 3; ++i) {
        int startRet = imuManager.StartCollection();
        EXPECT_EQ(startRet, 0);
        EXPECT_EQ(imuManager.GetIMUManagerStatus(), static_cast<uint8_t>(IMU_STATE::COLLECT));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        int suspendRet = imuManager.Suspend();
        EXPECT_EQ(suspendRet, 0);
        EXPECT_EQ(imuManager.GetIMUManagerStatus(), static_cast<uint8_t>(IMU_STATE::IDLE));
    }

    imuManager.StopLoop();
    if (imuThread.joinable()) {
        imuThread.join();
    }

    // Final log file (from last StartCollection) must have a valid header
    std::ifstream logFile("test_imu_cycles.csv");
    ASSERT_TRUE(logFile.is_open());
    std::string header;
    ASSERT_TRUE(std::getline(logFile, header));
    EXPECT_EQ(header, "Timestamp_ms, Gyro_X_dps, Gyro_Y_dps, Gyro_Z_dps, Mag_X_uT, Mag_Y_uT, Mag_Z_uT, Temperature_C");
    logFile.close();
    std::remove("test_imu_cycles.csv");
}

// TODO: Error handling tests