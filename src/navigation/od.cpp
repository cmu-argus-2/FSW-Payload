#include "navigation/od.hpp"
#include "spdlog/spdlog.h"
#include "toml.hpp"
#include <stdexcept>


INIT_config::INIT_config()
: 
collection_period(10),
target_samples(20), 
max_collection_time(3600), // 1 hr
max_downtime_for_restart(60) // 1hr
{
}

BATCH_OPT_config::BATCH_OPT_config()
: 
solver_function_tolerance(1e-6),
solver_parameter_tolerance(1e-10),
max_iterations(10000),
bias_mode(BIAS_MODE::FIX_BIAS),
max_dt(60.0),
compute_covariance(false),
use_j2(false),
use_drag(false),
bc_inv_nominal(0.0),
bc_inv_std(1e-8),
integrator(Integrator::EULER)
{
}

OD_Config::OD_Config()
{
}



OD::OD(const std::string& config_path)
{
    SPDLOG_INFO("Will read OD configuration file...");
    ReadConfig(config_path);
}

OD::~OD()
{
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
    config.init.max_downtime_for_restart = INIT_params->get_as<int64_t>("max_downtime_for_restart_in_minutes")->value_or(config.init.max_downtime_for_restart);

    // BATCH_OPT
    auto BATCH_OPT_params = params["BATCH_OPT"].as_table();
    config.batch_opt.solver_function_tolerance = get_param_as_double(BATCH_OPT_params, "solver_function_tolerance", config.batch_opt.solver_function_tolerance);
    config.batch_opt.solver_parameter_tolerance = get_param_as_double(BATCH_OPT_params, "solver_parameter_tolerance", config.batch_opt.solver_parameter_tolerance);
    config.batch_opt.max_iterations = BATCH_OPT_params->get_as<int64_t>("max_iterations")->value_or(config.batch_opt.max_iterations);
    config.batch_opt.max_dt = get_param_as_double(BATCH_OPT_params, "max_dt", config.batch_opt.max_dt);
    config.batch_opt.bias_mode = static_cast<BIAS_MODE>(BATCH_OPT_params->get_as<int64_t>("bias_mode")->value_or(static_cast<int64_t>(config.batch_opt.bias_mode)));
    config.batch_opt.compute_covariance = BATCH_OPT_params->get_as<bool>("compute_covariance")->value_or(config.batch_opt.compute_covariance);
    config.batch_opt.use_j2   = BATCH_OPT_params->get_as<bool>("use_j2")->value_or(config.batch_opt.use_j2);
    config.batch_opt.use_drag = BATCH_OPT_params->get_as<bool>("use_drag")->value_or(config.batch_opt.use_drag);
    config.batch_opt.bc_inv_nominal = get_param_as_double(BATCH_OPT_params, "bc_inv_nominal", config.batch_opt.bc_inv_nominal);
    config.batch_opt.bc_inv_std     = get_param_as_double(BATCH_OPT_params, "bc_inv_std",     config.batch_opt.bc_inv_std);
    config.batch_opt.integrator     = static_cast<Integrator>(BATCH_OPT_params->get_as<int64_t>("integrator")->value_or(static_cast<int64_t>(config.batch_opt.integrator)));
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
    SPDLOG_INFO("  solver_function_tolerance: {}", config.batch_opt.solver_function_tolerance);
    SPDLOG_INFO("  solver_parameter_tolerance: {}", config.batch_opt.solver_parameter_tolerance);
    SPDLOG_INFO("  max_iterations: {}", config.batch_opt.max_iterations);
    SPDLOG_INFO("  max_dt: {}", config.batch_opt.max_dt);
    SPDLOG_INFO("  bias_mode: {}", static_cast<int>(config.batch_opt.bias_mode));
    SPDLOG_INFO("  compute_covariance: {}", config.batch_opt.compute_covariance);
    SPDLOG_INFO("  use_j2: {}", config.batch_opt.use_j2);
    SPDLOG_INFO("  use_drag: {}", config.batch_opt.use_drag);
    SPDLOG_INFO("  bc_inv_nominal: {}", config.batch_opt.bc_inv_nominal);
    SPDLOG_INFO("  bc_inv_std: {}", config.batch_opt.bc_inv_std);
    SPDLOG_INFO("  integrator: {} ({})", static_cast<int>(config.batch_opt.integrator),
                config.batch_opt.integrator == Integrator::RK4 ? "RK4" : "Euler");

}

// TODO: To be removed once run batch opt is updated to use the OD class
OD_Config ReadODConfig(const std::string& config_path)
{
    toml::table params;
    try
    {
        params = toml::parse_file(config_path);
    }
    catch (const toml::parse_error& err)
    { 
        SPDLOG_ERROR("Failed to parse config file: {}", err.what());
        return OD_Config{};
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
    
    OD_Config od_config;

    // INIT
    auto INIT_params = params["INIT"].as_table();
    od_config.init.collection_period = INIT_params->get_as<int64_t>("collection_period")->value_or(od_config.init.collection_period);
    od_config.init.target_samples = INIT_params->get_as<int64_t>("target_samples")->value_or(od_config.init.target_samples);
    od_config.init.max_collection_time = INIT_params->get_as<int64_t>("max_collection_time")->value_or(od_config.init.max_collection_time);
    od_config.init.max_downtime_for_restart = INIT_params->get_as<int64_t>("max_downtime_for_restart_in_minutes")->value_or(od_config.init.max_downtime_for_restart);

    // BATCH_OPT
    auto BATCH_OPT_params = params["BATCH_OPT"].as_table();
    od_config.batch_opt.solver_parameter_tolerance = get_param_as_double(BATCH_OPT_params, "solver_parameter_tolerance", od_config.batch_opt.solver_parameter_tolerance);
    od_config.batch_opt.solver_function_tolerance = get_param_as_double(BATCH_OPT_params, "solver_function_tolerance", od_config.batch_opt.solver_function_tolerance);
    od_config.batch_opt.max_iterations = BATCH_OPT_params->get_as<int64_t>("max_iterations")->value_or(od_config.batch_opt.max_iterations);
    od_config.batch_opt.max_dt = get_param_as_double(BATCH_OPT_params, "max_dt", od_config.batch_opt.max_dt);
    od_config.batch_opt.bias_mode = static_cast<BIAS_MODE>(BATCH_OPT_params->get_as<int64_t>("bias_mode")->value_or(static_cast<int64_t>(od_config.batch_opt.bias_mode)));
    od_config.batch_opt.compute_covariance = BATCH_OPT_params->get_as<bool>("compute_covariance")->value_or(od_config.batch_opt.compute_covariance);
    od_config.batch_opt.use_j2   = BATCH_OPT_params->get_as<bool>("use_j2")->value_or(od_config.batch_opt.use_j2);
    od_config.batch_opt.use_drag = BATCH_OPT_params->get_as<bool>("use_drag")->value_or(od_config.batch_opt.use_drag);
    od_config.batch_opt.bc_inv_nominal = get_param_as_double(BATCH_OPT_params, "bc_inv_nominal", od_config.batch_opt.bc_inv_nominal);
    od_config.batch_opt.bc_inv_std     = get_param_as_double(BATCH_OPT_params, "bc_inv_std",     od_config.batch_opt.bc_inv_std);
    od_config.batch_opt.integrator     = static_cast<Integrator>(BATCH_OPT_params->get_as<int64_t>("integrator")->value_or(static_cast<int64_t>(od_config.batch_opt.integrator)));

    // Print the configuration
    SPDLOG_INFO("OD Configuration parameters set");

    return od_config;
}
