/*
Test script of IMUManager class
Author: Pedro Cachim
*/
#include "imu/imu_manager.hpp"

#include "spdlog/spdlog.h"
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

int main(int argc, char** argv) {

    // parse duration from argv (seconds), default 60
    int duration_sec = 10*60; // 3600*5;
    if (argc > 1) {
        try {
            duration_sec = std::stoi(argv[1]);
            if (duration_sec < 0) duration_sec = 0;
        } catch (...) {
            spdlog::error("Invalid test duration '{}', using 60 seconds", argv[1]);
            duration_sec = 60;
        }
    }
    std::string log_file = "./data/datasets/gyro_log_" + std::to_string(std::time(nullptr)) + ".csv";
    if (argc > 2) {
        log_file = argv[2];

        try {
            fs::path p(log_file);
            if (fs::exists(p) && fs::is_directory(p)) {
                // If user passed a directory, treat it as destination folder and append filename
                p /= "gyro_log.csv";
                log_file = p.string();
            }
        } catch (const std::exception& e) {
            spdlog::error("Filesystem check failed for log file path '{}': {}", log_file, e.what());
        }

    }
    
    std::cout << "Test duration: " << duration_sec << " seconds\n";
    
    std::cout << "Starting BMX160 gyro-only test..." << std::endl;

    IMUManager imuManager;

    // Create RunLoop thread
    std::thread imu_thread = std::thread(&IMUManager::RunLoop, &imuManager);

    // Set the logging file path for the IMU manager
    imuManager.SetLogFile(log_file);
    imuManager.SetSampleRate(25.0f);

    // Start the collection loop 
    imuManager.StartCollection();
    auto start_time = std::chrono::high_resolution_clock::now();

    uint8_t pmu_status;
    if (imuManager.ReadPowerModeStatus(&pmu_status) == 0) {
        spdlog::info("IMU Power Mode Status while collecting: 0x{:02X}", pmu_status);
    } else {
        spdlog::error("Failed to read IMU power mode status while collecting");
    }

    // After duration, stop the loop and exit
    while (std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::high_resolution_clock::now() - start_time)
               .count() < duration_sec) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    imuManager.Suspend();

    // Test getting the telemetry data in suspended mode
    spdlog::info("IMU Manager status after suspension: {}", imuManager.GetIMUManagerStatus());
    float temp_after_suspend;
    if (imuManager.ReadTemperatureData(&temp_after_suspend) == 0) {
        spdlog::info("IMU Temperature after suspension: {}", temp_after_suspend);
    } else {
        spdlog::error("Failed to read IMU temperature after suspension");
    }
    if (imuManager.ReadPowerModeStatus(&pmu_status) == 0) {
        spdlog::info("IMU Power Mode Status after suspension: 0x{:02X}", pmu_status);
    } else {
        spdlog::error("Failed to read IMU power mode status after suspension");
    }

    // exit
    imuManager.StopLoop();
    if (imu_thread.joinable()) {
        imu_thread.join();
    }
    
    return 0;
}
