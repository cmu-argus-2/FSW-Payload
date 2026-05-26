/*
  run_od_on_dataset <dataset_folder>
                    [--od-config <path>]
                    [--system-config <path>]
                    [--use-j2 | --no-use-j2]
                    [--use-drag | --no-use-drag]
                    [--compute-covariance | --no-compute-covariance]
                    [--bias-mode <0|1|2>]
                    [--integrator <0|1>]
                    [--max-run-time <sec>]
                    [--out <out_path>]

  Runs navigation OD on a previously captured dataset.

  Config resolution order (highest wins):
    1. Struct defaults in od.cpp
    2. --od-config file (fields present in the file override defaults)
    3. Explicit CLI flags (override file and defaults)

  If --od-config is omitted, config/od.toml is tried; if absent, defaults are used.

  --out  File to write the generated results directory path into. Falls back
         to path.out if not provided or not writable.
*/

#include "navigation/od.hpp"

#include <fstream>
#include <optional>
#include <string>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

static constexpr const char* kDefaultOutPath = "path.out";

static std::string ResolveOutPath(const std::string& path)
{
    if (path == kDefaultOutPath) return path;
    std::ofstream f(path, std::ios::app);
    if (f.is_open()) return path;
    spdlog::warn("Cannot write to '{}', using default output '{}'", path, kDefaultOutPath);
    return kDefaultOutPath;
}

static void WriteResult(const std::string& out_file, const std::string& content)
{
    std::ofstream f(out_file, std::ios::trunc);
    if (!f.is_open()) { spdlog::error("Failed to write result to '{}'", out_file); return; }
    f << content << '\n';
}

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::info);

    CLI::App app{"Run navigation OD on a captured dataset."};
    app.allow_extras(false);

    // ── Positional ────────────────────────────────────────────────────────────
    std::string dataset_folder;
    app.add_option("dataset_folder", dataset_folder, "Path to the dataset folder")->required();

    // ── Path options ──────────────────────────────────────────────────────────
    std::string od_config_path     = OD_DEFAULT_CONFIG_PATH;
    std::string system_config_path = "config/config.toml";
    std::string out_path           = kDefaultOutPath;

    CLI::Option* opt_config_path = app.add_option("--od-config",     od_config_path,     "OD config TOML (default: " + std::string(OD_DEFAULT_CONFIG_PATH) + ")");
    CLI::Option* opt_system_config_path = app.add_option("--system-config", system_config_path, "System config TOML (default: config/config.toml)");
    CLI::Option* opt_out_path = app.add_option("--out",           out_path,           "File to write the results directory path into");

    // ── batch_opt overrides (all optional) ───────────────────────────────────
    // std::optional lets us distinguish "not provided" from "explicitly set".
    std::optional<bool>   use_j2;
    std::optional<bool>   use_drag;
    std::optional<bool>   compute_covariance;
    std::optional<int>    bias_mode_int;
    std::optional<int>    integrator_int;
    std::optional<int>    max_iterations;
    std::optional<double> solver_function_tolerance;
    std::optional<double> solver_parameter_tolerance;
    std::optional<double> max_run_time_sec;
    // double cd_nominal; // could configure these too, probbly wont. Drag is low at 600
    // double cd_std;

    CLI::Option* opt_use_j2 = app.add_flag("--use-j2,--no-use-j2{false}",
                 use_j2,
                 "Enable J2 gravity perturbation (default: on)");
    CLI::Option* opt_use_drag = app.add_flag("--use-drag,--no-use-drag{false}",
                 use_drag,
                 "Enable atmospheric drag (default: off)");
    CLI::Option* opt_compute_covariance = app.add_flag("--compute-covariance,--no-compute-covariance{false}",
                 compute_covariance,
                 "Compute output covariance (default: off)");
    CLI::Option* opt_bias_mode = app.add_option("--bias-mode",    bias_mode_int,    "Gyro bias mode: 0=none 1=fixed 2=time-varying")
        ->check(CLI::Range(0, 2));
    CLI::Option* opt_integrator = app.add_option("--integrator",   integrator_int,   "Orbit integrator: 0=Euler 1=RK4")
        ->check(CLI::Range(0, 1));
    CLI::Option* opt_max_run_time = app.add_option("--max-run-time", max_run_time_sec, "Solver wall-clock time cap (seconds)");
    CLI::Option* opt_solver_function_tolerance = app.add_option("--solver-function-tolerance", solver_function_tolerance, "Solver function tolerance");
    CLI::Option* opt_solver_parameter_tolerance = app.add_option("--solver-parameter-tolerance", solver_parameter_tolerance, "Solver parameter tolerance");
    CLI::Option* opt_max_iterations = app.add_option("--max-iterations", max_iterations, "Solver maximum iterations");

    CLI11_PARSE(app, argc, argv);

    // ── Build OD config: defaults → file → CLI ────────────────────────────────
    const ODConfigResult file_result = ReadODConfig(od_config_path);
    if (file_result.code != ErrorCode::OK) {
        spdlog::error("Failed to load OD config from '{}'", od_config_path);
        return 1;
    }
    OD_Config od_config = file_result.config;

    if (opt_use_j2->count() > 0) {
        if (use_j2.has_value())             od_config.batch_opt.use_j2             = *use_j2;
    }

    if (opt_use_drag->count() > 0) {
        if (use_drag.has_value())           od_config.batch_opt.use_drag           = *use_drag;
    }

    if (opt_compute_covariance->count() > 0) {
        if (compute_covariance.has_value()) od_config.batch_opt.compute_covariance = *compute_covariance;
    }

    if (opt_bias_mode->count() > 0) {
        if (bias_mode_int.has_value())      od_config.batch_opt.bias_mode          = static_cast<BIAS_MODE>(*bias_mode_int);
    }

    if (opt_integrator->count() > 0) {
        if (integrator_int.has_value())     od_config.batch_opt.integrator         = static_cast<Integrator>(*integrator_int);
    }

    if (opt_max_run_time->count() > 0) {
        if (max_run_time_sec.has_value())   od_config.batch_opt.max_run_time_sec   = *max_run_time_sec;
    }

    if (opt_solver_function_tolerance->count() > 0) {
        if (solver_function_tolerance.has_value())   od_config.batch_opt.solver_function_tolerance   = *solver_function_tolerance;
    }

    if (opt_solver_parameter_tolerance->count() > 0) {
        if (solver_parameter_tolerance.has_value())   od_config.batch_opt.solver_parameter_tolerance   = *solver_parameter_tolerance;
    }

    if (opt_max_iterations->count() > 0) {
        if (max_iterations.has_value())   od_config.batch_opt.max_iterations   = *max_iterations;
    }

    spdlog::info("Batch optimization config:\n"
                 "  use_j2 = {}\n"
                 "  use_drag = {}\n"
                 "  compute_covariance = {}\n"
                 "  bias_mode = {}\n"
                 "  integrator = {}\n"
                 "  max_run_time_sec = {}\n"
                 "  solver_function_tolerance = {}\n"
                 "  solver_parameter_tolerance = {}\n"
                 "  max_iterations = {}\n",
                 od_config.batch_opt.use_j2,
                 od_config.batch_opt.use_drag,
                 od_config.batch_opt.compute_covariance,
                 static_cast<int>(od_config.batch_opt.bias_mode),
                 static_cast<int>(od_config.batch_opt.integrator),
                 od_config.batch_opt.max_run_time_sec,
                 od_config.batch_opt.solver_function_tolerance,
                 od_config.batch_opt.solver_parameter_tolerance,
                 od_config.batch_opt.max_iterations);

    // ── Run ───────────────────────────────────────────────────────────────────
    ODRequest request;
    request.dataset_folder     = dataset_folder;
    request.od_config_path     = od_config_path;
    request.system_config_path = system_config_path;
    request.od_config_override = od_config;

    const ODResult result = RunODOnDataset(request);
    spdlog::info("OD complete. Results in {}", result.results_dir);
    WriteResult(ResolveOutPath(out_path), result.results_dir);
    if (result.code != ErrorCode::OK) {
        spdlog::error("OD pipeline failed at stage {} with error code {}.",
                      static_cast<int>(result.stage), static_cast<int>(result.code));
        return 1;
    }

    return 0;
}
