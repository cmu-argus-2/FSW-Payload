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

    spdlog::info("Waiting for cameras to stabilize...");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    spdlog::info("Capturing single frame per camera");
    cam_manager.CaptureFrames();

    spdlog::info("Waiting for frame capture to complete...");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    spdlog::info("Saving frames...");
    uint8_t saved = cam_manager.SaveLatestFrames();
    spdlog::info("Saved {} frame(s).", saved);

    spdlog::info("Disabling cameras...");
    std::array<bool, NUM_CAMERAS> disabled;
    int disabled_count = cam_manager.DisableCameras(disabled);
    spdlog::info("Cameras disabled: {}", disabled_count);

    return 0;
}
