#include "navigation/od.hpp"
#include "payload.hpp"
#include "spdlog/spdlog.h"


OD::OD()
: 
process_state(OD_STATE::IDLE)
{
}




void OD::RunLoop(Payload* payload)
{
    loop_flag = true;

    while (loop_flag.load())
    {
        {
            std::unique_lock<std::mutex> lock(mtx_active);
            cv_active.wait(lock, [this] { return !loop_flag.load() || process_state != OD_STATE::IDLE; });
            SPDLOG_DEBUG("OD loop - Waking up");
        }

        if (!loop_flag.load())
        {
            break;
        }

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

    SPDLOG_INFO("OD Loop Stopped");
}

void OD::StopLoop()
{
    {
        std::lock_guard<std::mutex> lock(mtx_active);
        loop_flag.store(false);
    }
    cv_active.notify_all(); 
}


OD_STATE OD::GetState() const
{
    return process_state;
}

void OD::StartExperiment()
{
    experiment_done.store(false);
    process_state.store(OD_STATE::INIT);
}

bool OD::IsExperimentDone() const
{
    return experiment_done.load();
}

bool OD::PingRunningStatus()
{
    return loop_flag.load();
}



void OD::_Initialize(Payload* payload)
{
    // Initialize the OD process
    // TODO: for now
    process_state.store(OD_STATE::BATCH_OPT);
}

void OD::_DoBatchOptimization(Payload* payload)
{
    // Perform batch optimization
    // TODO
    process_state.store(OD_STATE::TRACKING);
}


void OD::_DoTracking(Payload* payload)
{
    // Perform tracking
    // TODO
    process_state.store(OD_STATE::IDLE);
}

