#include "navigation/od.hpp"
#include "payload.hpp"
#include "spdlog/spdlog.h"

OD::OD()
: 
process_state(OD_STATE::IDLE),
_experiment_done(false)
{
}




void OD::RunLoop(Payload* payload)
{

    switch (process_state)
    {
        case OD_STATE::IDLE:
        {
            SPDLOG_INFO("OD: IDLE");
            break;
        }

        case OD_STATE::INIT:
        {
            SPDLOG_INFO("OD: INIT");
            _Initialize(payload);
            break;
        }

        case OD_STATE::BATCH_OPT:
        {
            SPDLOG_INFO("OD: BATCH_OPT");
            _DoBatchOptimization(payload);
            break;
        }

        case OD_STATE::TRACKING:
        {
            SPDLOG_INFO("OD: TRACKING");
            _DoTracking(payload);
            break;
        }

        default:
            SPDLOG_WARN("OD: Unknown process state");
            break;
    }


}

void OD::StartExperiment()
{
    _experiment_done.store(false);
    process_state = OD_STATE::INIT;
}

void OD::IsExperimentDone() const
{
    return _experiment_done.load();
}





void OD::_Initialize(Payload* payload)
{
    // Initialize the OD process
    // TODO: for now
    process_state = OD_STATE::BATCH_OPT;
}

void OD::_DoBatchOptimization(Payload* payload)
{
    // Perform batch optimization
    // TODO
    process_state = OD_STATE::TRACKING;
}


void OD::_DoTracking(Payload* payload)
{
    // Perform tracking
    // TODO
    process_state = OD_STATE::IDLE;
}

