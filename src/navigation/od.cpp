#include "navigation/od.hpp"
#include "payload.hpp"
#include "spdlog/spdlog.h"
#include "toml.hpp"


INIT_config::INIT_config()
: 
collection_period(10),
target_samples(20),
max_collection_time(3600)
{
}

BATCH_OPT_config::BATCH_OPT_config()
: 
tolerance_solver(0.05),
max_iterations(10000)
{
}

TRACKING_config::TRACKING_config()
: 
gyro_update_frequency(10.0),
img_update_frequency(1.0)
{
}



OD::OD()
: 
process_state(OD_STATE::IDLE),
dataset_collector(nullptr)
{
}




void OD::RunLoop()
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
                _Initialize();
                break;
            }

            case OD_STATE::BATCH_OPT:
            {
                SPDLOG_INFO("OD: BATCH_OPT");
                _DoBatchOptimization();
                break;
            }

            case OD_STATE::TRACKING:
            {
                SPDLOG_INFO("OD: TRACKING");
                _DoTracking();
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

void OD::ReadConfig(std::string_view config_path)
{
    
    toml::table config = toml::parse_file(config_path);

    auto INIT_config = config["INIT"].as_table();
    auto BATCH_OPT_config = config["BATCH_OPT"].as_table();
    auto TRACKING_config = config["TRACKING"].as_table();


}


void OD::_Initialize()
{
    // Initialize the OD process
    // TODO: for now
    process_state.store(OD_STATE::BATCH_OPT);
}

void OD::_DoBatchOptimization()
{
    // Perform batch optimization
    // TODO
    process_state.store(OD_STATE::TRACKING);
}


void OD::_DoTracking()
{
    // Perform tracking
    // TODO
    process_state.store(OD_STATE::IDLE);
}

