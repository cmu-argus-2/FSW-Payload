/*
Test script to read gyro data from BMX160 IMU at 1 Hz.
Author: Pedro Cachim (with the aid of ChatGPT)
*/
#include "imu/bmx160.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>

int main() {
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

    while (true) {
        int32_t ret = imu.getSensorXYZ(gyroData, BMI160::DPS_125);
        if (ret != BMI160::RTN_NO_ERROR) {
            std::cerr << "getSensorXYZ (gyro) failed, code " << ret << "\n";
        } else {
            std::cout << "Gyro [dps]: "
                      << gyroData.xAxis.scaled << ", "
                      << gyroData.yAxis.scaled << ", "
                      << gyroData.zAxis.scaled << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1)); // 1 Hz print
    }

    return 0;
}
