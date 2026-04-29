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

    app.add_option("--od-config",     od_config_path,     "OD config TOML (default: " + std::string(OD_DEFAULT_CONFIG_PATH) + ")");
    app.add_option("--system-config", system_config_path, "System config TOML (default: config/config.toml)");
    app.add_option("--out",           out_path,           "File to write the results directory path into");

    // ── batch_opt overrides (all optional) ───────────────────────────────────
    // std::optional lets us distinguish "not provided" from "explicitly set".
    std::optional<bool>   use_j2;
    std::optional<bool>   use_drag;
    std::optional<bool>   compute_covariance;
    std::optional<int>    bias_mode_int;
    std::optional<int>    integrator_int;
    std::optional<double> max_run_time_sec;

    app.add_flag("--use-j2,--no-use-j2",
                 use_j2,
                 "Enable J2 gravity perturbation (default: on)");
    app.add_flag("--use-drag,--no-use-drag",
                 use_drag,
                 "Enable atmospheric drag (default: off)");
    app.add_flag("--compute-covariance,--no-compute-covariance",
                 compute_covariance,
                 "Compute output covariance (default: off)");
    app.add_option("--bias-mode",    bias_mode_int,    "Gyro bias mode: 0=none 1=fixed 2=time-varying")
        ->check(CLI::Range(0, 2));
    app.add_option("--integrator",   integrator_int,   "Orbit integrator: 0=Euler 1=RK4")
        ->check(CLI::Range(0, 1));
    app.add_option("--max-run-time", max_run_time_sec, "Solver wall-clock time cap (seconds)");

    CLI11_PARSE(app, argc, argv);

    // ── Build OD config: defaults → file → CLI ────────────────────────────────
    const ODConfigResult file_result = ReadODConfig(od_config_path);
    if (file_result.code != ErrorCode::OK) {
        spdlog::error("Failed to load OD config from '{}'", od_config_path);
        return 1;
    }
    OD_Config od_config = file_result.config;

    if (use_j2.has_value())             od_config.batch_opt.use_j2             = *use_j2;
    if (use_drag.has_value())           od_config.batch_opt.use_drag           = *use_drag;
    if (compute_covariance.has_value()) od_config.batch_opt.compute_covariance = *compute_covariance;
    if (bias_mode_int.has_value())      od_config.batch_opt.bias_mode          = static_cast<BIAS_MODE>(*bias_mode_int);
    if (integrator_int.has_value())     od_config.batch_opt.integrator         = static_cast<Integrator>(*integrator_int);
    if (max_run_time_sec.has_value())   od_config.batch_opt.max_run_time_sec   = *max_run_time_sec;

    // ── Run ───────────────────────────────────────────────────────────────────
    ODRequest request;
    request.dataset_folder     = dataset_folder;
    request.od_config_path     = od_config_path;
    request.system_config_path = system_config_path;
    request.od_config_override = od_config;

    const ODResult result = RunODOnDataset(request);
    if (result.code != ErrorCode::OK) {
        spdlog::error("OD pipeline failed at stage {} with error code {}.",
                      static_cast<int>(result.stage), static_cast<int>(result.code));
        return 1;
    }

    spdlog::info("OD complete. Results in {}", result.results_dir);
    WriteResult(ResolveOutPath(out_path), result.results_dir);
    return 0;
}
