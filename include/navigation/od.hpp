#ifndef OD_HPP
#define OD_HPP

#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <atomic>

enum class OD_STATE : uint8_t 
{
    IDLE = 0, // No operation, waiting for command to start
    INIT = 1, // Initialize the OD process by periodically capturing and storing frames. Each frame is being filtered for regions of interest and landmarks are identified.
    BATCH_OPT = 2, // Perform batch optimization on the stored landmarks
    TRACKING = 3 // Once a good initial state is obtained, the system will continue tracking in real-time landmarks from frames and perform state updates.
};



class OD
{

public:

    OD();


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

    void _Initialize();
    void _DoBatchOptimization();
    void _DoTracking();



    
};




#endif // OD_HPP