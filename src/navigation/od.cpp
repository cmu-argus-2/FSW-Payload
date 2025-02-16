#include <optional>
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
gyro_update_frequency(10.0f),
img_update_frequency(1.0f)
{
}

OD_Config::OD_Config()
{
}



OD::OD(const std::string& config_path)
: 
process_state(OD_STATE::IDLE),
dataset_collector(nullptr)
{
    SPDLOG_INFO("Will read OD configuration file...");
    ReadConfig(config_path);
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

void OD::ReadConfig(const std::string& config_path)
{
    toml::table params;
    try
    {
        params = toml::parse_file(config_path);
    }
    catch (const toml::parse_error& err)
    { 
        SPDLOG_ERROR("Failed to parse config file: {}", err.what());
        return;
    }

    // Helper to get parameter as double (supports int64_t and double)
    auto get_param_as_double = [](const toml::table* params, const std::string& key, double default_value) -> double {
        if (auto val = params->get_as<double>(key)) {
            return **val;
        }
        if (auto val_int = params->get_as<int64_t>(key)) {
            return static_cast<double>(**val_int);
        }
        return default_value;
    };
    
    // INIT
    auto INIT_params = params["INIT"].as_table();
    config.init.collection_period = INIT_params->get_as<int64_t>("collection_period")->value_or(config.init.collection_period);
    config.init.target_samples = INIT_params->get_as<int64_t>("target_samples")->value_or(config.init.target_samples);
    config.init.max_collection_time = INIT_params->get_as<int64_t>("max_collection_time")->value_or(config.init.max_collection_time);

    // BATCH_OPT
    auto BATCH_OPT_params = params["BATCH_OPT"].as_table();
    config.batch_opt.tolerance_solver = get_param_as_double(BATCH_OPT_params, "tolerance_solver", config.batch_opt.tolerance_solver);
    config.batch_opt.max_iterations = BATCH_OPT_params->get_as<int64_t>("max_iterations")->value_or(config.batch_opt.max_iterations);

    // TRACKING
    auto TRACKING_params = params["TRACKING"].as_table();
    config.tracking.gyro_update_frequency = get_param_as_double(TRACKING_params, "gyro_update_frequency", config.tracking.gyro_update_frequency);
    config.tracking.img_update_frequency = get_param_as_double(TRACKING_params, "img_update_frequency", config.tracking.img_update_frequency);

    // TODO: safe value checking on each params
    
    LogConfig();

}


void OD::LogConfig()
{
    SPDLOG_INFO("OD Current Configuration parameters: ");

    SPDLOG_INFO("INIT_config:");
    SPDLOG_INFO("  collection_period: {}", config.init.collection_period);
    SPDLOG_INFO("  target_samples: {}", config.init.target_samples);
    SPDLOG_INFO("  max_collection_time: {}", config.init.max_collection_time);

    SPDLOG_INFO("BATCH_OPT_config:");
    SPDLOG_INFO("  tolerance_solver: {}", config.batch_opt.tolerance_solver);
    SPDLOG_INFO("  max_iterations: {}", config.batch_opt.max_iterations);

    SPDLOG_INFO("TRACKING_config:");
    SPDLOG_INFO("  gyro_update_frequency: {}", config.tracking.gyro_update_frequency);
    SPDLOG_INFO("  img_update_frequency: {}", config.tracking.img_update_frequency);
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

