/*
  run_od_on_dataset <dataset_folder> [od_config_path] [system_config_path] [--out <out_path>]

  Runs navigation OD on a previously captured dataset. Landmark measurements
  are prepared automatically if the dataset has processed LDNet frame outputs.

    --out  File to write the generated results directory path into. Falls back
           to path.out if not provided or not writable.
*/

#include "navigation/od.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

static constexpr const char* kDefaultODConfigPath = "config/od.toml";
static constexpr const char* kDefaultOutPath      = "path.out";

// Returns the value of --flag, or default_val if not present.
static std::string GetFlag(int argc, char** argv, const char* flag, const char* default_val = "")
{
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == flag) return argv[i + 1];
    return default_val;
}

// Returns positional args, skipping any --flag <value> pairs.
static std::vector<std::string> GetPositional(int argc, char** argv)
{
    std::vector<std::string> result;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.size() > 2 && a[0] == '-' && a[1] == '-') { ++i; continue; }
        result.push_back(a);
    }
    return result;
}

// Returns the provided path if writable, otherwise falls back to kDefaultOutPath.
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
    const auto positional = GetPositional(argc, argv);
    if (positional.empty()) {
        std::cerr << "Usage: run_od_on_dataset <dataset_folder> "
                     "[od_config_path] [system_config_path] [--out <out_path>]\n";
        return 1;
    }
    spdlog::set_level(spdlog::level::info);

    const std::string out_path = ResolveOutPath(
        GetFlag(argc, argv, "--out", kDefaultOutPath));

    ODRequest request;
    request.dataset_folder = positional[0];
    request.od_config_path = positional.size() > 1 ? positional[1] : kDefaultODConfigPath;
    if (positional.size() > 2) request.system_config_path = positional[2];

    const ODResult result = RunODOnDataset(request);
    if (result.code != ErrorCode::OK) {
        spdlog::error("OD pipeline failed at stage {} with error code {}.",
                      static_cast<int>(result.stage), static_cast<int>(result.code));
        return 1;
    }

    spdlog::info("OD complete. Results in {}", result.results_dir);
    WriteResult(out_path, result.results_dir);
    return 0;
}
