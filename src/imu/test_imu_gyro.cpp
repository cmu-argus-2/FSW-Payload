/*
Test script to read gyro data from BMX160 IMU at 1 Hz.
Author: Pedro Cachim (with the aid of ChatGPT)
*/
#include "imu/bmx160.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {

    // parse duration from argv (seconds), default 60
    int duration_sec = 60;
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
    
    std::cout << "Starting BMX160 gyro-only test (1 Hz)..." << std::endl;

    // 1. Construct IMU on /dev/i2c-7, address 0x68 (SDO low)
    BMI160_I2C imu("/dev/i2c-7", BMI160_I2C::I2C_ADRS_SDO_LO);

    // 2. Power modes: gyro ON, accel & mag OFF (suspend)
    if (imu.setSensorPowerMode(BMI160::GYRO, BMI160::NORMAL) != BMI160::RTN_NO_ERROR) {
        std::cerr << "Failed to set gyro power mode to NORMAL\n";
        return 1;
    }

    imu.setSensorPowerMode(BMI160::ACC, BMI160::SUSPEND);
    imu.setSensorPowerMode(BMI160::MAG, BMI160::SUSPEND);
    
    // gyro startup delay
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    // 3. Configure gyro: range=±125 dps, BW=normal, ODR=25 Hz
    BMI160::GyroConfig gcfg = BMI160::DEFAULT_GYRO_CONFIG;

    gcfg.range = BMI160::DPS_125;     // ±125 dps
    gcfg.bwp   = BMI160::GYRO_BWP_2;  // normal mode
    gcfg.odr   = BMI160::GYRO_ODR_6;  // 25 Hz

    if (imu.setSensorConfig(gcfg) != BMI160::RTN_NO_ERROR) {
        std::cerr << "Failed to configure gyro\n";
        return 1;
    }

    std::cout << "Gyro configured: range=±125 dps, BW=normal, ODR=25 Hz\n";

    // 4. Read gyro every second
    BMI160::SensorData gyroData;

    std::ofstream ofs(log_file, std::ios::app);
    if (!ofs) {
        std::cerr << "Failed to open " << log_file << " for writing\n";
        return 1;
    } else {
        std::cout << "Logging gyro data to " << log_file << "\n";
    }

    auto start_time = std::chrono::steady_clock::now();
    auto last_point = std::chrono::steady_clock::now();
    auto one_sec = std::chrono::seconds(1);
    auto now_point = std::chrono::steady_clock::now();
    auto elapsed = now_point - last_point;
    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now() - start_time)
               .count() < duration_sec) {
        int32_t ret = imu.getSensorXYZ(gyroData, BMI160::DPS_125);
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);

        if (ret != BMI160::RTN_NO_ERROR) {
            std::cerr << "getSensorXYZ (gyro) failed, code " << ret << "\n";
            ofs << std::put_time(std::localtime(&t), "%F %T") << " ERROR " << ret << '\n';
        } else {
            std::cout << "Gyro [dps]: "
                      << gyroData.xAxis.scaled << ", "
                      << gyroData.yAxis.scaled << ", "
                      << gyroData.zAxis.scaled << std::endl;

            ofs << std::put_time(std::localtime(&t), "%F %T") << ' '
                << std::fixed << std::setprecision(6)
                << gyroData.xAxis.scaled << ' '
                << gyroData.yAxis.scaled << ' '
                << gyroData.zAxis.scaled << '\n';
        }

        ofs.flush();
        // ensure at least 1 second has passed since the last time we reached this point
        now_point = std::chrono::steady_clock::now();
        elapsed = now_point - last_point;
        
        if (elapsed < one_sec) {
            std::this_thread::sleep_for(one_sec - elapsed);
        }
        last_point = std::chrono::steady_clock::now();
    }

    return 0;
}
