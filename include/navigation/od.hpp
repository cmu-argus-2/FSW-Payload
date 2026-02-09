#ifndef OD_HPP
#define OD_HPP

#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vision/dataset.hpp>

#define OD_DEFAULT_COLLECTION_PERIOD 10
#define OD_DEFAULT_COLLECTION_SAMPLES 30
#define DATASET_KEY_OD "OD"
#define OD_DEFAULT_CONFIG_PATH "config/od.toml"

enum class OD_STATE : uint8_t 
{
    IDLE = 0, // No operation, waiting for command to start
    INIT = 1, // Initialize the OD process by periodically capturing and storing frames. Each frame is being filtered for regions of interest and landmarks are identified.
    BATCH_OPT = 2, // Perform batch optimization on the stored landmarks
};

enum class BIAS_MODE : uint8_t
{
    NO_BIAS  = 0,  // no gyro bias estimation
    FIX_BIAS = 1, // fixed/constant gyro bias estimation
    TV_BIAS  = 2   // time-varying gyro bias estimation
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
    BIAS_MODE bias_mode;
    double max_dt;
    BATCH_OPT_config();
};

struct OD_Config
{
    INIT_config init;
    BATCH_OPT_config batch_opt;

    OD_Config();
};

class OD
{

public:

    OD(const std::string& config_path = OD_DEFAULT_CONFIG_PATH);

    // Main running loop for the OD process
    void RunLoop();
    void StopLoop();

    OD_STATE GetState() const;
    void StartExperiment();
    bool IsExperimentDone() const;



private:

    std::atomic<OD_STATE> process_state;
    std::atomic<bool> experiment_done = false;
    std::atomic<bool> loop_flag = false;
    std::mutex mtx_active;
    std::condition_variable cv_active;

    void SwitchState(OD_STATE new_od_state);


    // Configurations
    OD_Config config;

    // Read the config yaml file 
    void ReadConfig(const std::string& config_path);
    void LogConfig();

    std::shared_ptr<DatasetManager> dataset_collector;

    // Check if the OD is running
    bool PingRunningStatus(); 

    /* 
    Must be called frequently within the DoXXX function process so the OD process can stop properly and save correctly its states for the next run
    - Return True if states have been saved and the process must stop 
    Example usage: 
        if (HandleStop())
        {
            return;
        }
    */
    bool HandleStop();

    // Main running steps for each stages
    void _DoInit();
    void _DoBatchOptimization();


};

OD_Config ReadODConfig(const std::string& config_path);



#endif // OD_HPP