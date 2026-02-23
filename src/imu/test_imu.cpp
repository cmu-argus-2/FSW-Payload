/*
Test script to read gyro data from BMX160 IMU at 1 Hz.
Author: Pedro Cachim (with the aid of ChatGPT)
*/
#include "imu/imu_manager.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <vector>
#include <cmath>
#include <thread>

namespace fs = std::filesystem;
/*
void variance_loop(int duration_sec, BMI160& imu, BMI160::GyroConfig& gcfg, float sample_rate_hz) {

    BMI160::SensorData gyroData;
    BMI160::SensorData magData;
    float temperature;
    BMI160::SensorTime sensorTime;

    auto start_time = std::chrono::steady_clock::now();
    auto last_point = std::chrono::steady_clock::now();
    auto now_point = std::chrono::steady_clock::now();
    auto elapsed = now_point - last_point;
    auto samp_period = std::chrono::duration<double>(1.0 / sample_rate_hz);

    std::vector<BMI160::SensorData> gyro_samples;
    
    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start_time)
               .count() < duration_sec) {
        int32_t ret = imu.getSensorXYZandSensorTime(gyroData, sensorTime, gcfg.range);
        std::cout << "Time: " << sensorTime.seconds << " s, "
                  << "Gyro [dps]: " << gyroData.xAxis.scaled << ", "
                  << gyroData.yAxis.scaled << ", "
                  << gyroData.zAxis.scaled << std::endl;
        
        gyro_samples.push_back(gyroData);
        
        // ensure at least sampling period has passed
        now_point = std::chrono::steady_clock::now();
        elapsed = now_point - last_point;
        
        if (elapsed < samp_period) {
            std::this_thread::sleep_for(samp_period - elapsed);
        }
        last_point = std::chrono::steady_clock::now();
    }
    
    // Calculate mean and standard deviation
    double mean_x = 0, mean_y = 0, mean_z = 0;
    for (const auto& sample : gyro_samples) {
        mean_x += sample.xAxis.scaled;
        mean_y += sample.yAxis.scaled;
        mean_z += sample.zAxis.scaled;
    }
    mean_x /= gyro_samples.size();
    mean_y /= gyro_samples.size();
    mean_z /= gyro_samples.size();
    
    double std_x = 0, std_y = 0, std_z = 0;
    for (const auto& sample : gyro_samples) {
        std_x += (sample.xAxis.scaled - mean_x) * (sample.xAxis.scaled - mean_x);
        std_y += (sample.yAxis.scaled - mean_y) * (sample.yAxis.scaled - mean_y);
        std_z += (sample.zAxis.scaled - mean_z) * (sample.zAxis.scaled - mean_z);
    }
    std_x = std::sqrt(std_x / gyro_samples.size());
    std_y = std::sqrt(std_y / gyro_samples.size());
    std_z = std::sqrt(std_z / gyro_samples.size());
    
    std::cout << "\nGyro Statistics:\n"
              << "X: mean=" << mean_x << " dps, std=" << std_x << " dps\n"
              << "Y: mean=" << mean_y << " dps, std=" << std_y << " dps\n"
              << "Z: mean=" << mean_z << " dps, std=" << std_z << " dps\n";
}

void collection_loop(int duration_sec, BMI160& imu, BMI160::GyroConfig& gcfg, float sample_rate_hz) {
    
    BMI160::SensorData gyroData;
    BMI160::SensorData magData;
    float temperature;
    BMI160::SensorTime sensorTime;

    std::ofstream ofs(log_file, std::ios::app);
    if (!ofs) {
        std::cerr << "Failed to open " << log_file << " for writing\n";
        return 1;
    } else {
        std::cout << "Logging gyro data to " << log_file << "\n";
    }

    auto start_time = std::chrono::steady_clock::now();
    auto last_point = std::chrono::steady_clock::now();
    auto now_point = std::chrono::steady_clock::now();
    auto elapsed = now_point - last_point;
    auto samp_period = std::chrono::duration<double>(1.0 / sample_rate_hz);

    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start_time)
               .count() < duration_sec) {
        // int32_t ret = imu.getSensorXYZ(gyroData, BMI160::DPS_125);
        // alternative
        int32_t ret = imu.getSensorXYZandSensorTime(gyroData, sensorTime, gcfg.range);
        std::cout << "SensorTime Gyro: " << sensorTime.seconds << " seconds\n";
        int32_t ret_mag = imu.getMagSensorXYZ(magData);
        imu.getTemperature(&temperature);

        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);

        if (ret != BMI160::RTN_NO_ERROR && ret_mag != BMI160::RTN_NO_ERROR) {
            std::cerr << "getSensorXYZ (gyro) failed, code " << ret << "\n";
            ofs << std::put_time(std::localtime(&t), "%F %T") << " ERROR " << ret << '\n';
        } else {
            std::cout << "Gyro [dps]: "
                      << gyroData.xAxis.scaled << ", "
                      << gyroData.yAxis.scaled << ", "
                      << gyroData.zAxis.scaled << ", "
                      << "Mag [µT]: "
                      << magData.xAxis.scaled << ", "
                      << magData.yAxis.scaled << ", "
                      << magData.zAxis.scaled << ", "
                      << "Temperature [C]: "
                      << temperature << std::endl;

            ofs << std::put_time(std::localtime(&t), "%F %T") << ' '
                << sensorTime.seconds << ' '
                << std::fixed << std::setprecision(6)
                << gyroData.xAxis.scaled << ' '
                << gyroData.yAxis.scaled << ' '
                << gyroData.zAxis.scaled << ' '
                << magData.xAxis.scaled << ' '
                << magData.yAxis.scaled << ' '
                << magData.zAxis.scaled << ' '
                << temperature << '\n';
        }

        ofs.flush();
        // ensure at least 1 second has passed since the last time we reached this point
        now_point = std::chrono::steady_clock::now();
        elapsed = now_point - last_point;
        
        if (elapsed < samp_period) {
            std::this_thread::sleep_for(samp_period - elapsed);
        }
        last_point = std::chrono::steady_clock::now();
    }
    ofs.close();

}
*/

int main(int argc, char** argv) {

    // parse duration from argv (seconds), default 60
    int duration_sec = 10;
    if (argc > 1) {
        try {
            duration_sec = std::stoi(argv[1]);
            if (duration_sec < 0) duration_sec = 0;
        } catch (...) {
            std::cerr << "Invalid test duration '" << argv[1] << "', using 60 seconds\n";
            duration_sec = 60;
        }
    }
    std::string log_file = "./data/datasets/gyro_log.txt";
    if (argc > 2) {
        log_file = argv[2];

        try {
            fs::path p(log_file);
            if (fs::exists(p) && fs::is_directory(p)) {
                // If user passed a directory, treat it as destination folder and append filename
                p /= "gyro_log.txt";
                log_file = p.string();
            }
        } catch (const std::exception& e) {
            std::cerr << "Filesystem check failed: " << e.what() << '\n';
        }

    }
    
    std::cout << "Test duration: " << duration_sec << " seconds\n";
    
    std::cout << "Starting BMX160 gyro-only test..." << std::endl;

    IMUManager imuManager;

    // Create RunLoop thread
    std::thread imu_thread = std::thread(&IMUManager::RunLoop, &imuManager);

    // Start the collection loop 
    imuManager.StartCollection();
    auto start_time = std::chrono::high_resolution_clock::now();

    // After duration, stop the loop and exit
    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::high_resolution_clock::now() - start_time)
               .count() < duration_sec) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    imuManager.Suspend();

    // Test getting the telemetry data in suspended mode

    // exit
    imuManager.StopLoop();
    if (imu_thread.joinable()) {
        imu_thread.join();
    }
    /*
    // 1. Construct IMU on /dev/i2c-7, address 0x68 (SDO low)
    BMI160_I2C imu("/dev/i2c-7", BMI160_I2C::I2C_ADRS_SDO_LO);

    // 2. Power modes: gyro ON, accel & mag OFF (suspend)
    if (imu.setSensorPowerMode(BMI160::GYRO, BMI160::NORMAL) != BMI160::RTN_NO_ERROR) {
        std::cerr << "Failed to set gyro power mode to NORMAL\n";
        return 1;
    }

    imu.setSensorPowerMode(BMI160::ACC, BMI160::SUSPEND);
    imu.setSensorPowerMode(BMI160::MAG, BMI160::NORMAL);
    
    // gyro startup delay
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    // 3. Configure gyro: range=±125 dps, BW=normal, ODR=25 Hz
    BMI160::GyroConfig gcfg = BMI160::DEFAULT_GYRO_CONFIG;

    gcfg.range = BMI160::DPS_125;     // ±125 dps
    gcfg.bwp   = BMI160::GYRO_BWP_0;  // OSR2
    gcfg.odr   = BMI160::GYRO_ODR_6;  // 25 Hz

    if (imu.setSensorConfig(gcfg) != BMI160::RTN_NO_ERROR) {
        std::cerr << "Failed to configure gyro\n";
        // IMU status would be set to error
        return 1;
    }
    imu.setMagnConf();

    imu.getSensorConfig(gcfg);

    std::cout << "Gyro configured: range=" << gcfg.range << ", BW=" << gcfg.bwp << ", ODR=" << gcfg.odr << " \n";

    float sample_rate_hz = 25.0f; // 25 Hz sampling

    variance_loop(duration_sec, imu, gcfg, sample_rate_hz);

    // Loop and Log
    // collection_loop(duration_sec, imu, gcfg, sample_rate_hz);

    // Suspend mode log
    */
    return 0;
}
