/**/
#include <imu/bmx160.hpp>
#include <string>
#include <atomic>
#include <core/timing.hpp>
// TODO: recheck for unnecessary includes
#include "spdlog/spdlog.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <filesystem>

enum class IMU_STATE : uint8_t 
{
    IDLE = 0, // Suspended mode
    COLLECT = 1, // Sucessfully initialized and collecting data
    ERROR_DEVICE = 2,    // Error state with operational IMU
    ERROR_DEVICE_NOT_FOUND = 3 // Device not found
};

constexpr std::string_view GetIMUState(IMU_STATE state) {
    switch (state) {
        case IMU_STATE::IDLE: return "IDLE";
        case IMU_STATE::COLLECT: return "COLLECT";
        case IMU_STATE::ERROR_DEVICE: return "ERROR_DEVICE";
        case IMU_STATE::ERROR_DEVICE_NOT_FOUND: return "ERROR_DEVICE_NOT_FOUND";
        default: return "UNKNOWN";
    }
}

struct IMUData
{

    float gyro_x_dps;
    float gyro_y_dps;
    float gyro_z_dps;
    float mag_x_uT;
    float mag_y_uT;
    float mag_z_uT;
    float temperature_C;
};

// Need to get a full log of IMU errors

// IMUManager Class
class IMUManager
{
    public:
        IMUManager();
        ~IMUManager();

        // setters
        void SetSampleRate(float rate_hz);
        void SetLogFile(const std::string& file_path);

        // getters
        uint8_t GetIMUManagerStatus();
        float GetSampleRate() const { return sample_rate_hz; }
        std::string GetLogFile() const { return log_file; }

        // IMU Manager main loop control
        void RunLoop();
        void StopLoop();
        
        // IMU Manager state control
        int StartCollection();
        int Suspend();

        // Read from Sensors
        int32_t ReadSensorData(BMI160::SensorData &gyroData, BMI160::SensorData &magData, float *temperature);
        int32_t ReadGyroData(BMI160::SensorData &gyroData, BMI160::GyroConfig gcfg= BMI160::DEFAULT_GYRO_CONFIG);
        int32_t ReadMagnetometerData(BMI160::SensorData &magData);
        int32_t ReadTemperatureData(float *temperature);
        int32_t ReadPowerModeStatus(uint8_t *pmuStatus);
        
    private:
        // Should only be used for debugging, not intended for regular verification
        int32_t ReadErrorStatus(uint8_t *errReg);
        // May become useful internally for debugging
        int32_t ReadSensorStatus(bool *gyrSelfTestOk, bool *magManOp, bool *focRdy, bool *nvmRdy, bool *drdyMag, bool *drdyGyr, bool *drdyAcc);
        // Data Logging
        void LogSensorData(uint64_t timestamp, const BMI160::SensorData &gyroData, const BMI160::SensorData &magData, float temperature);

        BMI160_I2C bmi160;

        std::atomic<IMU_STATE> state;
        float sample_rate_hz= 1.0f; // default sample rate
        std::atomic<bool> loop_flag = false; // flag to control the main loop
        // Latest sensor data pointers
        // TODO: thread safety for these data if they are accessed from multiple threads, consider using mutex or atomic types
        std::string log_file = "imu_log.csv"; // default log file path
        std::ofstream ofs; // output file stream for logging

};