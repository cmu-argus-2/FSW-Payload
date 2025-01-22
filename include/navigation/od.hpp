#ifndef OD_HPP
#define OD_HPP

#include <cstdint>
#include <condition_variable>
#include <atomic>

// Forward declaration of Payload
class Payload;

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



    void RunLoop(Payload* payload);

    void StartExperiment();

    bool IsExperimentDone() const;
    


private:

    OD_STATE process_state;

    std::condition_variable cv_active;

    void _Initialize(Payload* payload);
    void _DoBatchOptimization(Payload* payload);
    void _DoTracking(Payload* payload);

    

    std::atomic<bool> _experiment_done;




};




#endif // OD_HPP