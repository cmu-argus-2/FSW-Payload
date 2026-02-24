/**/
// task to get and collect timestamped data from the imu sensor
#include <imu/imu_manager.hpp>

IMUManager::IMUManager() : bmi160("/dev/i2c-7", BMI160_I2C::I2C_ADRS_SDO_LO), state(IMU_STATE::IDLE) {
    // Initialize BMI160 with I2C interface, default address
    if (!bmi160.deviceFound()) {
        SPDLOG_ERROR("BMX160 IMU not found on I2C bus");
        state.store(IMU_STATE::ERROR_DEVICE_NOT_FOUND);
    } else {
        SPDLOG_INFO("BMX160 IMU found and initialized");
        Suspend(); // Ensure sensors are in suspend mode on initialization
    }
    
}

IMUManager::~IMUManager() {
}

void IMUManager::SetSampleRate(float rate_hz) {
    sample_rate_hz = rate_hz;
}

void IMUManager::SetLogFile(const std::string& file_path) {
    log_file = file_path;
}

uint8_t IMUManager::GetIMUManagerStatus() {
    return static_cast<uint8_t>(state.load()); // Return current state of the IMU
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
    uint8_t errReg;
    int32_t ret;
    uint64_t timestamp;
    
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
                timestamp = timing::GetCurrentTimeMs();
                ret = ReadSensorData(gyroData, magData, &temperature);
                if (ret != BMI160::RTN_NO_ERROR) {
                    SPDLOG_ERROR("Failed to read sensor data during collection, code {}", ret);
                    ofs << timestamp << " ms, ERROR " << ret << '\n';
                } else {
                    LogSensorData(timestamp, gyroData, magData, temperature);
                }
                break;
            }

            case IMU_STATE::ERROR_DEVICE:
            {
                SPDLOG_ERROR("IMU Manager in ERROR state, suspending for now");
                ReadErrorStatus(&errReg);
                // TODO: Error handling for the IMU manager
                Suspend(); // Put sensors in low power mode
                // loop_flag.store(false);
                break;
            }

            case IMU_STATE::ERROR_DEVICE_NOT_FOUND:
            {
                // For now, loop is still kept running but doing nothing.
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
    if (state.load() == IMU_STATE::ERROR_DEVICE_NOT_FOUND) {
        SPDLOG_ERROR("Cannot start collection, IMU device not found");
        return 1; // Error, device not found
    }

    // 1. Set Sensor Power modes
    if (bmi160.setSensorPowerMode(BMI160::GYRO, BMI160::NORMAL) != BMI160::RTN_NO_ERROR) {
        state.store(IMU_STATE::ERROR_DEVICE); // Update state to error
        SPDLOG_ERROR("Failed to set gyro power mode to NORMAL");
        SPDLOG_INFO("IMU Manager status set to: {}", GetIMUState(state.load()));
        return 1; // Error setting gyro power mode
    }

    bmi160.setSensorPowerMode(BMI160::ACC, BMI160::SUSPEND);
    // Optional: enable mag if needed
    bmi160.setSensorPowerMode(BMI160::MAG, BMI160::NORMAL);
    
    // gyro startup delay
    // if gyro is in suspend, it takes 55ms, if its in fast start up mode, it takes 10 ms
    // when not used, gyro is in fast start up mode
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    
    // 2. Configure gyro: range=±125 dps, BW=OSR2, ODR=25 Hz
    if (bmi160.setSensorConfig(BMI160::DEFAULT_GYRO_CONFIG) != BMI160::RTN_NO_ERROR) {
        SPDLOG_ERROR("Failed to configure gyro");
        state.store(IMU_STATE::ERROR_DEVICE); // Update state to error
        SPDLOG_INFO("IMU Manager status set to: {}", GetIMUState(state.load()));
        return 1;
    }
    bmi160.setMagnConf();

    // 3. Open log file for writing
    ofs.open(log_file, std::ios::out | std::ios::trunc);
    if (!ofs) {
        SPDLOG_ERROR("Failed to open {} for writing", log_file);
        state.store(IMU_STATE::ERROR_DEVICE); // Update state to error
        SPDLOG_INFO("IMU Manager status set to: {}", GetIMUState(state.load()));
        return 1;
    } else {
        SPDLOG_INFO("Logging gyro data to {}", log_file);
        ofs << "Timestamp_ms, Gyro_X_dps, Gyro_Y_dps, Gyro_Z_dps, Mag_X_uT, Mag_Y_uT, Mag_Z_uT, Temperature_C\n";
    }
    
    state.store(IMU_STATE::COLLECT); // Update state to collecting
    SPDLOG_INFO("IMU Manager status set to: {}", GetIMUState(state.load()));

    return 0; // Success
}

int IMUManager::Suspend() {
    if (state.load() == IMU_STATE::ERROR_DEVICE_NOT_FOUND) {
        SPDLOG_ERROR("Cannot suspend, IMU device not found");
        return 1; // Error, device not found
    }
    // Set all sensors to suspend mode but gyro to still have access to temperature data
    // gyro fast start up mode saves power and still allows for power consumption
    bmi160.setSensorPowerMode(BMI160::GYRO, BMI160::FAST_START_UP);
    bmi160.setSensorPowerMode(BMI160::ACC, BMI160::SUSPEND);
    bmi160.setSensorPowerMode(BMI160::MAG, BMI160::SUSPEND);

    // Close file logging stream if open
    ofs.close();
    
    state.store(IMU_STATE::IDLE); // Update state to idle
    SPDLOG_INFO("IMU Manager status set to: {}", GetIMUState(state.load()));

    // if there is data to be read
    int32_t ret;
    bool gyrSelfTestOk, magManOp, focRdy, nvmRdy, drdyMag, drdyGyr, drdyAcc;
    ret = ReadSensorStatus(&gyrSelfTestOk, &magManOp, &focRdy, &nvmRdy, &drdyMag, &drdyGyr, &drdyAcc);
    if (ret != BMI160::RTN_NO_ERROR) {
        SPDLOG_ERROR("Failed to read sensor status during suspend, code {}", ret);
        return 1; // Error reading sensor status
    } 
    
    if (drdyMag) {
        BMI160::SensorData magData;
        ReadMagnetometerData(magData); // Clear mag data if available
    }

    if (drdyGyr) {
        BMI160::SensorData gyroData;
        ReadGyroData(gyroData); // Clear gyro data if available
    }
    // check again if it worked
    ret = ReadSensorStatus(&gyrSelfTestOk, &magManOp, &focRdy, &nvmRdy, &drdyMag, &drdyGyr, &drdyAcc);

    if (ret != BMI160::RTN_NO_ERROR) {
        SPDLOG_ERROR("Failed to read sensor status during suspend verification, code {}", ret);
        return 1; // Error reading sensor status
    }

    if (drdyMag || drdyGyr) {
        SPDLOG_ERROR("Failed to suspend sensors, data still ready - Mag DRDY: {}, Gyro DRDY: {}", drdyMag, drdyGyr);
        return 1; // Error, sensors not properly suspended
    } else {
        SPDLOG_INFO("IMU successfully suspended, no data ready");
    }

    return 0; // Success
}

int32_t IMUManager::ReadSensorData(BMI160::SensorData &gyroData, BMI160::SensorData &magData, float *temperature) {
    int32_t ret_gyro = ReadGyroData(gyroData);
    int32_t ret_mag = ReadMagnetometerData(magData);
    int32_t ret_temp = ReadTemperatureData(temperature);
    int32_t ret = (ret_gyro != BMI160::RTN_NO_ERROR) ? ret_gyro : ((ret_mag != BMI160::RTN_NO_ERROR) ? ret_mag : ret_temp);
    if (ret != BMI160::RTN_NO_ERROR) {
        SPDLOG_ERROR("Failed to read sensor data - Gyro: {}, Mag: {}, Temp: {}", ret_gyro, ret_mag, ret_temp);
        state.store(IMU_STATE::ERROR_DEVICE); // Update state to error if any of the data retrievals failed
        SPDLOG_INFO("IMU Manager status set to: {}", GetIMUState(state.load()));
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
    if (state.load() == IMU_STATE::ERROR_DEVICE_NOT_FOUND) {
        *temperature = 0.0f; // cannot read temperature, dummy value
        return 0; // cannot read temperature
    }
    return bmi160.getTemperature(temperature);
}

int32_t IMUManager::ReadErrorStatus(uint8_t *errReg)
{
    int32_t rtnVal = bmi160.getErrorStatus(errReg);
    bool fatalError;
    BMI160::ErrorCodes errorCode;
    bool dropCmdError;
    bool i2cFailError;
    bool magDrdyError;
    bmi160.decodeErrorStatus(*errReg, &fatalError, &errorCode, &dropCmdError, &i2cFailError, &magDrdyError);
    SPDLOG_INFO("Decoded Error Status - Fatal Error: {}, Error Code: {}, Drop Cmd Error: {}, I2C Fail Error: {}, Mag DRDY Error: {}",
                 fatalError, bmi160.GetErrorCode(static_cast<BMI160::ErrorCodes>(errorCode)), dropCmdError, i2cFailError, magDrdyError);
    
    return rtnVal;

}

int32_t IMUManager::ReadPowerModeStatus(uint8_t *pmuStatus)
{
    int32_t rtnVal = bmi160.getPowerModeStatus(pmuStatus);
    BMI160::PowerModes magStatus, gyroStatus, accStatus;
    bmi160.decodePowerModeStatus(*pmuStatus, &magStatus, &gyroStatus, &accStatus);
    SPDLOG_INFO("Decoded Power Mode Status - Mag Status: {}, Gyro Status: {}, Acc Status: {}",
                 bmi160.GetPowerMode(magStatus), bmi160.GetPowerMode(gyroStatus), bmi160.GetPowerMode(accStatus));
    return rtnVal;
}

int32_t IMUManager::ReadSensorStatus(bool *gyrSelfTestOk, bool *magManOp, bool *focRdy, bool *nvmRdy, bool *drdyMag, bool *drdyGyr, bool *drdyAcc)
{
    uint8_t sensorStatus;
    int32_t rtnVal = bmi160.getSensorStatus(&sensorStatus);
    BMI160::decodeSensorStatus(sensorStatus, gyrSelfTestOk, magManOp, focRdy, nvmRdy, drdyMag, drdyGyr, drdyAcc);
    // SPDLOG_INFO("Decoded Sensor Status - Gyro Self Test OK: {}, Mag Man Op: {}, FOC Ready: {}, NVM Ready: {}, Mag DRDY: {}, Gyro DRDY: {}, Acc DRDY: {}",
    //              *gyrSelfTestOk, *magManOp, *focRdy, *nvmRdy, *drdyMag, *drdyGyr, *drdyAcc);
    return rtnVal;
}

void IMUManager::LogSensorData(uint64_t timestamp, const BMI160::SensorData &gyroData, const BMI160::SensorData &magData, float temperature) {
    ofs << timestamp << ','
        << std::fixed << std::setprecision(6)
        << gyroData.xAxis.scaled << ','
        << gyroData.yAxis.scaled << ','
        << gyroData.zAxis.scaled << ','
        << magData.xAxis.scaled << ','
        << magData.yAxis.scaled << ','
        << magData.zAxis.scaled << ','
        << temperature << '\n';

    ofs.flush();
}