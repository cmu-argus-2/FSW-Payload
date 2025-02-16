#ifndef OD_HPP
#define OD_HPP

#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vision/dataset.hpp>
#include <string_view>

#define OD_DEFAULT_COLLECTION_PERIOD 10
#define OD_DEFAULT_COLLECTION_SAMPLES 30

#define DATASET_KEY_OD "OD"

enum class OD_STATE : uint8_t 
{
    IDLE = 0, // No operation, waiting for command to start
    INIT = 1, // Initialize the OD process by periodically capturing and storing frames. Each frame is being filtered for regions of interest and landmarks are identified.
    BATCH_OPT = 2, // Perform batch optimization on the stored landmarks
    TRACKING = 3 // Once a good initial state is obtained, the system will continue tracking in real-time landmarks from frames and perform state updates.
};


struct INIT_config
{
    uint32_t collection_period;
    uint32_t target_samples;
    uint32_t max_collection_time;

    INIT_config();
};

struct BATCH_OPT_config
{
    double tolerance_solver;
    uint32_t max_iterations;

    BATCH_OPT_config();
};

struct TRACKING_config
{
    double gyro_update_frequency;
    double img_update_frequency;

    TRACKING_config();
};


class OD
{

public:

    OD();

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

    // This must be called frequently to check if the OD process must stop and save its states to disk
    bool PingRunningStatus(); 

    // Read the config yaml file 
    void ReadConfig(std::string_view config_path);


    std::shared_ptr<DatasetManager> dataset_collector;

    

    void _Initialize();
    void _DoBatchOptimization();
    void _DoTracking();



    
};




#endif // OD_HPP