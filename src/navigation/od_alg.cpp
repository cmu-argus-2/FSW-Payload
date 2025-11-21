#include "navigation/od.hpp"
#include "navigation/batch_optimization.hpp"


void OD::_DoInit()
{
    // Initialize the OD process

    // Might need to read an OD previous run file ~ something that logs the results 

    // Check if there is any active AND valid data collection already started (in case came from reboot/or any)
    // -- validity check can also include the amount of time since we did capture 
    // Also check that we haven't already used that guy 
    // If yes, restart that collection process
    // If No, create that new data process and start it
    // Monitor continuously progress 
    // If we need to stop --> stop collection and exit 
    // if completed, switch our state to BATCH_OPT



    SwitchState(OD_STATE::BATCH_OPT);
}

void OD::_DoBatchOptimization()
{
    // Perform batch optimization
    // TODO
    // Data verification check 
    // Minimum require to perform an estimation? 
    // If not, need to redo an INIT phase..
    // Set up optimization problem
    SwitchState(OD_STATE::IDLE);
}

