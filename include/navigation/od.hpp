#ifndef OD_HPP
#define OD_HPP

#include <cstdint>
#include <optional>
#include <string>
#include "navigation/od_measurements.hpp"
#include "vision/dataset.hpp"

// Forward declaration — full definition in configuration.hpp
struct CameraCalibration;
class CameraManager;
class IMUManager;
class InferenceManager;

#define OD_DEFAULT_COLLECTION_PERIOD 10
#define OD_DEFAULT_COLLECTION_SAMPLES 30
#define DATASET_KEY_OD "OD"
#define OD_DEFAULT_CONFIG_PATH "config/od.toml"

enum class ODStage : uint8_t
{
    DATASET_NOT_AVAILABLE = 0,
    DATASET_NOT_PROCESSED = 1,
    DATASET_PROCESSED = 2,
    MEASUREMENTS_READY = 3,
    INITIAL_GUESS_CREATED = 4,
    OD_COMPLETED = 5,
    FAILED = 6
};

enum class ODRunStatus : uint8_t
{
    NOT_STARTED = 0,
    RUNNING = 1,
    COMPLETED = 2,
    FAILED = 3,
    CANCELLED = 4
};

enum class BIAS_MODE : uint8_t
{
    NO_BIAS  = 0,  // no gyro bias estimation
    FIX_BIAS = 1, // fixed/constant gyro bias estimation
    TV_BIAS  = 2   // time-varying gyro bias estimation
};

enum Integrator : int
{
    EULER = 0,
    RK4   = 1
};


struct INIT_config
{
    uint32_t collection_period;
    uint32_t target_samples;
    uint32_t max_collection_time;
    uint32_t max_downtime_for_restart; 

    INIT_config();
};

struct BATCH_OPT_config
{
    double solver_function_tolerance;
    double solver_parameter_tolerance;
    uint32_t max_iterations;
    double max_run_time_sec;
    BIAS_MODE bias_mode;
    bool compute_covariance;
    bool use_j2;
    bool use_drag;
    double cd_nominal;
    double cd_std;
    Integrator integrator;
    BATCH_OPT_config();
};

struct OD_Config
{
    INIT_config init;
    BATCH_OPT_config batch_opt;

    OD_Config();
};

struct ODRequest
{
    std::string dataset_folder;
    std::string od_config_path = OD_DEFAULT_CONFIG_PATH;
    std::string system_config_path = "config/config.toml";
    // When set, takes precedence over od_config_path entirely.
    std::optional<OD_Config> od_config_override;
    DatasetConfig dataset_config = {
        OD_DEFAULT_COLLECTION_PERIOD,
        OD_DEFAULT_COLLECTION_SAMPLES,
        CAPTURE_MODE::PERIODIC_LDMK,
        0,
        IMU_COLLECTION_MODE::GYRO_ONLY,
        1,
        1.0f,
        ProcessingStage::LDNeted
    };
};

struct ODResult
{
    ErrorCode code = ErrorCode::OK;
    ODStage stage = ODStage::DATASET_NOT_AVAILABLE;
    std::string dataset_folder;
    std::string results_dir;
};

struct ODConfigResult
{
    ErrorCode code = ErrorCode::OK;
    OD_Config config;
};

struct ODMeasurementsResult
{
    ErrorCode code = ErrorCode::OK;
    ODMeasurements measurements;
};

class OD
{

public:

    OD(const std::string& config_path = OD_DEFAULT_CONFIG_PATH);
    ~OD();

    // Quick pre-check: does the dataset folder contain enough LDNeted frames with
    // landmarks and IMU data that spans the collection window?
    // Returns false (with reason logged) if OD is clearly infeasible.
    bool IsODPossible(const std::string& dataset_folder) const;

    // Convert a completed dataset folder into the measurement matrices required by the
    // batch optimizer. Infers the LD model version from the dataset frame metadata.
    // calibration  — shared intrinsics + per-camera cam_to_body rotation matrices
    // Returns ErrorCode::OK on success; otherwise returns a diagnostic error code.
    ErrorCode DatasetPrepare(const std::string& dataset_folder,
                             const CameraCalibration& calibration);

private:
    // Configurations
    OD_Config config;

    // Read the config yaml file 
    ErrorCode ReadConfig(const std::string& config_path);
    void LogConfig();

    ODMeasurements measurements_;
    bool measurements_ready_ = false;
};

ODConfigResult ReadODConfig(const std::string& config_path);
ODStage InspectDatasetForOD(const std::string& dataset_folder);
ODMeasurementsResult LoadODMeasurementsFromDataset(const std::string& dataset_folder);
ODResult RunODOnDataset(const ODRequest& request);
ODResult RunODPipeline(const ODRequest& request,
                       CameraManager& cam_manager,
                       IMUManager& imu_manager,
                       InferenceManager& inference_manager);



#endif // OD_HPP
