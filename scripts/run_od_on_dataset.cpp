/*
  run_od_on_dataset <dataset_folder> [od_config_path] [system_config_path]

  Runs navigation OD on a previously captured dataset. Landmark measurements
  are prepared automatically if the dataset has processed LDNet frame outputs.
*/

#include "navigation/od.hpp"

#include <iostream>
#include <string>

#include <spdlog/spdlog.h>

static constexpr const char* kDefaultODConfigPath = "config/od.toml";

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: run_od_on_dataset <dataset_folder> "
                     "[od_config_path] [system_config_path]\n";
        return 1;
    }
    spdlog::set_level(spdlog::level::info);

    ODRequest request;
    request.dataset_folder = argv[1];
    request.od_config_path = (argc > 2) ? argv[2] : kDefaultODConfigPath;
    if (argc > 3) request.system_config_path = argv[3];

    const ODResult result = RunODOnDataset(request);
    if (result.code != ErrorCode::OK) {
        spdlog::error("OD pipeline failed at stage {} with error code {}.",
                      static_cast<int>(result.stage), static_cast<int>(result.code));
        return 1;
    }

    spdlog::info("OD complete. Results in {}", result.results_dir);
    return 0;
}
