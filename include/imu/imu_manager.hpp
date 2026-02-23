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
    ERROR = 2    // Error state, check IMU status for details
};

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
        int32_t ReadErrorStatus(uint8_t *errReg);
        int32_t ReadPowerModeStatus(uint8_t *pmuStatus);
        int32_t ReadSensorStatus(uint8_t *sensorStatus);

        // Data Logging
        void LogSensorData(uint64_t timestamp, const BMI160::SensorData &gyroData, const BMI160::SensorData &magData, float temperature);

    private:
        BMI160_I2C bmi160;

        std::atomic<IMU_STATE> state;
        float sample_rate_hz= 1.0f; // default sample rate
        std::atomic<bool> loop_flag = false; // flag to control the main loop
        // Latest sensor data pointers
        // TODO: thread safety for these data if they are accessed from multiple threads, consider using mutex or atomic types
        std::string log_file = "imu_log.txt"; // default log file path
        std::ofstream ofs; // output file stream for logging

};