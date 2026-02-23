/**/
// task to get and collect timestamped data from the imu sensor
#include <imu/imu_manager.hpp>

IMUManager::IMUManager() : bmi160("/dev/i2c-7", BMI160_I2C::I2C_ADRS_SDO_LO), state(IMU_STATE::IDLE) {
    // Initialize BMI160 with I2C interface, default address
    Suspend(); // Ensure sensors are in suspend mode on initialization
}

IMUManager::~IMUManager() {
}

void IMUManager::RunLoop() {
    // based on camera manager
    loop_flag.store(true);
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_point = std::chrono::high_resolution_clock::now();
    auto now_point = std::chrono::high_resolution_clock::now();
    auto elapsed = now_point - last_point;
    auto samp_period = std::chrono::duration<double>(1.0 / sample_rate_hz);

    BMI160::SensorData gyroData;
    BMI160::SensorData magData;
    float temperature;
    
    while (loop_flag.load()) 
    {
        // Check for incoming commands (e.g. start/stop collection)
        switch (state.load())
        {
            case IMU_STATE::IDLE:
            {
                break;
            }

            case IMU_STATE::COLLECT:
            {
                auto timestamp = timing::GetCurrentTimeMs();
                int32_t ret = ReadSensorData(gyroData, magData, &temperature);
                if (ret != BMI160::RTN_NO_ERROR) {
                    std::cerr << "Read Sensor data failed, code " << ret << "\n";
                    ofs << timestamp << " ms, ERROR " << ret << '\n';
                } else {
                    LogSensorData(timestamp, gyroData, magData, temperature);
                }
                break;
            }

            case IMU_STATE::ERROR:
            {
                // TODO: General error handling for the IMU manager
                spdlog::error("IMU Manager in ERROR state, suspending for now");
                Suspend(); // Put sensors in low power mode
                // loop_flag.store(false);
                break;
            }
        }
        now_point = std::chrono::high_resolution_clock::now();
        elapsed = now_point - last_point;
        
        if (elapsed < samp_period) {
            std::this_thread::sleep_for(samp_period - elapsed);
        }
        last_point = std::chrono::high_resolution_clock::now();
    }
}

void IMUManager::StopLoop() {
    loop_flag.store(false);
}

int IMUManager::StartCollection() {

    // 1. Set Sensor Power modes
    if (bmi160.setSensorPowerMode(BMI160::GYRO, BMI160::NORMAL) != BMI160::RTN_NO_ERROR) {
        state = IMU_STATE::ERROR; // Update state to error
        return 1; // Error setting gyro power mode
    }

    bmi160.setSensorPowerMode(BMI160::ACC, BMI160::SUSPEND);
    // Optional: enable mag if needed
    bmi160.setSensorPowerMode(BMI160::MAG, BMI160::NORMAL);
    
    // gyro startup delay
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    
    // 2. Configure gyro: range=±125 dps, BW=OSR2, ODR=25 Hz
    if (bmi160.setSensorConfig(BMI160::DEFAULT_GYRO_CONFIG) != BMI160::RTN_NO_ERROR) {
        std::cerr << "Failed to configure gyro\n";
        state = IMU_STATE::ERROR; // Update state to error
        return 1;
    }
    bmi160.setMagnConf();

    // 3. Open log file for writing
    ofs.open(log_file, std::ios::app);
    if (!ofs) {
        spdlog::error("Failed to open {} for writing", log_file);
        state = IMU_STATE::ERROR; // Update state to error
        return 1;
    } else {
        spdlog::info("Logging gyro data to {}", log_file);
    }
    
    state = IMU_STATE::COLLECT; // Update state to collecting

    return 0; // Success
}

int IMUManager::Suspend() {
    // Set all sensors to suspend mode but gyro to still have access to temperature data
    // gyro fast start up mode saves power and still allows for power consumption
    bmi160.setSensorPowerMode(BMI160::GYRO, BMI160::FAST_START_UP);
    bmi160.setSensorPowerMode(BMI160::ACC, BMI160::SUSPEND);
    bmi160.setSensorPowerMode(BMI160::MAG, BMI160::SUSPEND);

    // Close file logging stream if open
    ofs.close();
    
    state = IMU_STATE::IDLE; // Update state to idle
    return 0; // Success
}

int32_t IMUManager::ReadSensorData(BMI160::SensorData &gyroData, BMI160::SensorData &magData, float *temperature) {
    int32_t ret_gyro = ReadGyroData(gyroData);
    int32_t ret_mag = ReadMagnetometerData(magData);
    int32_t ret_temp = ReadTemperatureData(temperature);
    int32_t ret = (ret_gyro != BMI160::RTN_NO_ERROR) ? ret_gyro : ((ret_mag != BMI160::RTN_NO_ERROR) ? ret_mag : ret_temp);
    if (ret != BMI160::RTN_NO_ERROR) {
        state = IMU_STATE::ERROR; // Update state to error if any of the data retrievals failed
    }
    return ret; // Return the result of the data retrievals
}

int32_t IMUManager::ReadGyroData(BMI160::SensorData &gyroData, BMI160::GyroConfig gcfg) {

    return bmi160.getSensorXYZ(gyroData, gcfg.range);
}

int32_t IMUManager::ReadMagnetometerData(BMI160::SensorData &magData) {
    return bmi160.getMagSensorXYZ(magData);
}

int32_t IMUManager::ReadTemperatureData(float *temperature) {
    return bmi160.getTemperature(temperature);
}

int32_t IMUManager::ReadErrorStatus(uint8_t *errReg)
{
    return bmi160.getErrorStatus(errReg);
}

int32_t IMUManager::ReadPowerModeStatus(uint8_t *pmuStatus)
{
    return bmi160.getPowerModeStatus(pmuStatus);
}

int32_t IMUManager::ReadSensorStatus(uint8_t *sensorStatus)
{
    return bmi160.getSensorStatus(sensorStatus);
}

uint8_t IMUManager::GetIMUManagerStatus() {
    return static_cast<uint8_t>(state.load()); // Return current state of the IMU
}


void IMUManager::LogSensorData(uint64_t timestamp, const BMI160::SensorData &gyroData, const BMI160::SensorData &magData, float temperature) {
    ofs << timestamp << " ms, "
        << std::fixed << std::setprecision(6)
        << gyroData.xAxis.scaled << ' '
        << gyroData.yAxis.scaled << ' '
        << gyroData.zAxis.scaled << ' '
        << magData.xAxis.scaled << ' '
        << magData.yAxis.scaled << ' '
        << magData.zAxis.scaled << ' '
        << temperature << '\n';

    ofs.flush();
}