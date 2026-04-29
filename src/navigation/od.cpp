#include "navigation/od.hpp"
#include "spdlog/spdlog.h"
#include "toml.hpp"
#include <filesystem>
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
max_run_time_sec(120.0),
bias_mode(BIAS_MODE::FIX_BIAS),
compute_covariance(false),
use_j2(true),
use_drag(false),
cd_nominal(2.2),
cd_std(1.0),
integrator(Integrator::EULER)
{
}

OD_Config::OD_Config()
{
}



OD::OD(const std::string& config_path)
{
    SPDLOG_INFO("Will read OD configuration file...");
    (void)ReadConfig(config_path);
}

OD::~OD()
{
}


ErrorCode OD::ReadConfig(const std::string& config_path)
{
    toml::table params;
    if (!std::filesystem::exists(config_path)) {
        SPDLOG_ERROR("OD config file does not exist: {}", config_path);
        return ErrorCode::FILE_DOES_NOT_EXIST;
    }

    try
    {
        params = toml::parse_file(config_path);
    }
    catch (const toml::parse_error& err)
    { 
        SPDLOG_ERROR("Failed to parse config file: {}", err.what());
        return ErrorCode::FILE_NOT_AVAILABLE;
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
    auto BATCH_OPT_params = params["BATCH_OPT"].as_table();
    if (!INIT_params || !BATCH_OPT_params) {
        SPDLOG_ERROR("OD config file missing required [INIT] or [BATCH_OPT] section: {}",
                     config_path);
        return ErrorCode::FILE_NOT_AVAILABLE;
    }
    config.init.collection_period = INIT_params->get_as<int64_t>("collection_period")->value_or(config.init.collection_period);
    config.init.target_samples = INIT_params->get_as<int64_t>("target_samples")->value_or(config.init.target_samples);
    config.init.max_collection_time = INIT_params->get_as<int64_t>("max_collection_time")->value_or(config.init.max_collection_time);
    config.init.max_downtime_for_restart = INIT_params->get_as<int64_t>("max_downtime_for_restart_in_minutes")->value_or(config.init.max_downtime_for_restart);

    // BATCH_OPT
    config.batch_opt.solver_function_tolerance = get_param_as_double(BATCH_OPT_params, "solver_function_tolerance", config.batch_opt.solver_function_tolerance);
    config.batch_opt.solver_parameter_tolerance = get_param_as_double(BATCH_OPT_params, "solver_parameter_tolerance", config.batch_opt.solver_parameter_tolerance);
    config.batch_opt.max_iterations = BATCH_OPT_params->get_as<int64_t>("max_iterations")->value_or(config.batch_opt.max_iterations);
    config.batch_opt.max_run_time_sec = get_param_as_double(BATCH_OPT_params, "max_run_time_sec", config.batch_opt.max_run_time_sec);
    config.batch_opt.bias_mode = static_cast<BIAS_MODE>(BATCH_OPT_params->get_as<int64_t>("bias_mode")->value_or(static_cast<int64_t>(config.batch_opt.bias_mode)));
    config.batch_opt.compute_covariance = BATCH_OPT_params->get_as<bool>("compute_covariance")->value_or(config.batch_opt.compute_covariance);
    config.batch_opt.use_j2   = BATCH_OPT_params->get_as<bool>("use_j2")->value_or(config.batch_opt.use_j2);
    config.batch_opt.use_drag = BATCH_OPT_params->get_as<bool>("use_drag")->value_or(config.batch_opt.use_drag);
    config.batch_opt.cd_nominal = get_param_as_double(BATCH_OPT_params, "cd_nominal", config.batch_opt.cd_nominal);
    config.batch_opt.cd_std     = get_param_as_double(BATCH_OPT_params, "cd_std",     config.batch_opt.cd_std);
    config.batch_opt.integrator     = static_cast<Integrator>(BATCH_OPT_params->get_as<int64_t>("integrator")->value_or(static_cast<int64_t>(config.batch_opt.integrator)));
    // TODO: safe value checking on each params
    
    LogConfig();
    return ErrorCode::OK;

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
    SPDLOG_INFO("  max_run_time_sec: {}", config.batch_opt.max_run_time_sec);
    SPDLOG_INFO("  bias_mode: {}", static_cast<int>(config.batch_opt.bias_mode));
    SPDLOG_INFO("  compute_covariance: {}", config.batch_opt.compute_covariance);
    SPDLOG_INFO("  use_j2: {}", config.batch_opt.use_j2);
    SPDLOG_INFO("  use_drag: {}", config.batch_opt.use_drag);
    SPDLOG_INFO("  cd_nominal: {}", config.batch_opt.cd_nominal);
    SPDLOG_INFO("  cd_std: {}", config.batch_opt.cd_std);
    SPDLOG_INFO("  integrator: {} ({})", static_cast<int>(config.batch_opt.integrator),
                config.batch_opt.integrator == Integrator::RK4 ? "RK4" : "Euler");

}

ODConfigResult ReadODConfig(const std::string& config_path)
{
    ODConfigResult result;
    toml::table params;
    if (!std::filesystem::exists(config_path)) {
        SPDLOG_WARN("OD config file not found, using defaults: {}", config_path);
        return result;
    }

    try
    {
        params = toml::parse_file(config_path);
    }
    catch (const toml::parse_error& err)
    {
        SPDLOG_ERROR("Failed to parse OD config file: {}", err.what());
        result.code = ErrorCode::FILE_NOT_AVAILABLE;
        return result;
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

    OD_Config& od_config = result.config;

    auto INIT_params      = params["INIT"].as_table();
    auto BATCH_OPT_params = params["BATCH_OPT"].as_table();
    if (!INIT_params)      SPDLOG_WARN("OD config missing [INIT] section, using defaults");
    if (!BATCH_OPT_params) SPDLOG_WARN("OD config missing [BATCH_OPT] section, using defaults");

    if (INIT_params) {
        od_config.init.collection_period        = INIT_params->get_as<int64_t>("collection_period")->value_or(od_config.init.collection_period);
        od_config.init.target_samples           = INIT_params->get_as<int64_t>("target_samples")->value_or(od_config.init.target_samples);
        od_config.init.max_collection_time      = INIT_params->get_as<int64_t>("max_collection_time")->value_or(od_config.init.max_collection_time);
        od_config.init.max_downtime_for_restart = INIT_params->get_as<int64_t>("max_downtime_for_restart_in_minutes")->value_or(od_config.init.max_downtime_for_restart);
    }

    if (BATCH_OPT_params) {
        od_config.batch_opt.solver_parameter_tolerance = get_param_as_double(BATCH_OPT_params, "solver_parameter_tolerance", od_config.batch_opt.solver_parameter_tolerance);
        od_config.batch_opt.solver_function_tolerance  = get_param_as_double(BATCH_OPT_params, "solver_function_tolerance",  od_config.batch_opt.solver_function_tolerance);
        od_config.batch_opt.max_iterations    = BATCH_OPT_params->get_as<int64_t>("max_iterations")->value_or(od_config.batch_opt.max_iterations);
        od_config.batch_opt.max_run_time_sec  = get_param_as_double(BATCH_OPT_params, "max_run_time_sec", od_config.batch_opt.max_run_time_sec);
        od_config.batch_opt.bias_mode         = static_cast<BIAS_MODE>(BATCH_OPT_params->get_as<int64_t>("bias_mode")->value_or(static_cast<int64_t>(od_config.batch_opt.bias_mode)));
        od_config.batch_opt.compute_covariance = BATCH_OPT_params->get_as<bool>("compute_covariance")->value_or(od_config.batch_opt.compute_covariance);
        od_config.batch_opt.use_j2            = BATCH_OPT_params->get_as<bool>("use_j2")->value_or(od_config.batch_opt.use_j2);
        od_config.batch_opt.use_drag          = BATCH_OPT_params->get_as<bool>("use_drag")->value_or(od_config.batch_opt.use_drag);
        od_config.batch_opt.cd_nominal        = get_param_as_double(BATCH_OPT_params, "cd_nominal", od_config.batch_opt.cd_nominal);
        od_config.batch_opt.cd_std            = get_param_as_double(BATCH_OPT_params, "cd_std",     od_config.batch_opt.cd_std);
        od_config.batch_opt.integrator        = static_cast<Integrator>(BATCH_OPT_params->get_as<int64_t>("integrator")->value_or(static_cast<int64_t>(od_config.batch_opt.integrator)));
    }

    // Print the configuration
    SPDLOG_INFO("OD Configuration parameters set");

    result.code = ErrorCode::OK;
    return result;
}
