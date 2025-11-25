#include <memory>
#include "spdlog/spdlog.h"
#include "vision/camera_manager.hpp"
#include "configuration.hpp"
#include "core/data_handling.hpp"

int main(int argc, char** argv)
{
    spdlog::info("Initializing camera test");

    auto config = std::make_unique<Configuration>();
    config->LoadConfiguration("config/config.toml");
    

    const auto& cam_configs = config->GetCameraConfigs();
    CameraManager cam_manager(cam_configs);

    spdlog::info("Enabling cameras...");
    std::array<bool, NUM_CAMERAS> activated;
    int count = cam_manager.EnableCameras(activated);
    spdlog::info("Cameras enabled: {}", count);

    spdlog::info("Capturing frames");
    cam_manager.CaptureFrames();

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    spdlog::info("Saving latest frames...");
    uint8_t saved = cam_manager.SaveLatestFrames();

    spdlog::info("Saved {} frame(s).", saved);

    return 0;
}
