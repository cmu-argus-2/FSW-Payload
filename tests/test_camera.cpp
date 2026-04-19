#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include "configuration.hpp"
#include "vision/camera.hpp"
#include "vision/camera_manager.hpp"
#include "inference/inference_manager.hpp"

// ── Helpers ───────────────────────────────────────────────────────────────────

// Minimum required sections so LoadConfiguration doesn't throw.
static constexpr const char* REQUIRED_SECTIONS = R"(
[camera-device.cam1]
id = 0
path = '/dev/video0'
resolution_width = 640
resolution_height = 480
enabled = true

[camera-device.cam2]
id = 1
path = '/dev/video1'
resolution_width = 640
resolution_height = 480
enabled = true

[camera-device.cam3]
id = 2
path = '/dev/video2'
resolution_width = 640
resolution_height = 480
enabled = true

[camera-device.cam4]
id = 3
path = '/dev/video3'
resolution_width = 640
resolution_height = 480
enabled = true

[imu-device]
chipid = 216
i2c_addr = 104
i2c_path = '/dev/i2c-7'
)";

static std::string write_temp_toml(const std::string& isp_section)
{
    std::string path = "/tmp/test_camera_isp_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + ".toml";
    std::ofstream f(path);
    f << REQUIRED_SECTIONS << "\n" << isp_section;
    return path;
}

// ── CameraISPConfig struct defaults ──────────────────────────────────────────

TEST(CameraISPConfigTest, StructDefaults)
{
    CameraISPConfig cfg;
    EXPECT_EQ(cfg.wbmode, 0);
    EXPECT_EQ(cfg.aelock, false);
    EXPECT_EQ(cfg.awblock, false);
    EXPECT_EQ(cfg.ee_mode, 1);
    EXPECT_FLOAT_EQ(cfg.ee_strength, -1.0f);
    EXPECT_EQ(cfg.aeantibanding, 1);
    EXPECT_FLOAT_EQ(cfg.exposurecompensation, 0.0f);
    EXPECT_EQ(cfg.tnr_mode, 1);
    EXPECT_FLOAT_EQ(cfg.tnr_strength, -1.0f);
    EXPECT_FLOAT_EQ(cfg.saturation, 1.0f);
    EXPECT_EQ(cfg.fps, DEFAULT_CAMERA_FPS);
    EXPECT_EQ(cfg.max_buffers, 2);
    EXPECT_FALSE(cfg.exposuretimerange.has_value());
    EXPECT_FALSE(cfg.gainrange.has_value());
    EXPECT_FALSE(cfg.ispdigitalgainrange.has_value());
}

// ── No [camera-isp] section → struct defaults preserved ──────────────────────

TEST(CameraISPConfigTest, NoISPSectionUsesStructDefaults)
{
    std::string path = write_temp_toml(""); // no [camera-isp] block
    Configuration cfg;
    cfg.LoadConfiguration(path);
    const CameraISPConfig& isp = cfg.GetCameraISPConfig();

    EXPECT_EQ(isp.wbmode, 0);
    EXPECT_EQ(isp.aelock, false);
    EXPECT_EQ(isp.awblock, false);
    EXPECT_EQ(isp.ee_mode, 1);
    EXPECT_FLOAT_EQ(isp.ee_strength, -1.0f);
    EXPECT_EQ(isp.aeantibanding, 1);
    EXPECT_FLOAT_EQ(isp.exposurecompensation, 0.0f);
    EXPECT_EQ(isp.tnr_mode, 1);
    EXPECT_FLOAT_EQ(isp.tnr_strength, -1.0f);
    EXPECT_FLOAT_EQ(isp.saturation, 1.0f);
    EXPECT_EQ(isp.fps, DEFAULT_CAMERA_FPS);
    EXPECT_EQ(isp.max_buffers, 2);
    EXPECT_FALSE(isp.exposuretimerange.has_value());
    EXPECT_FALSE(isp.gainrange.has_value());
    EXPECT_FALSE(isp.ispdigitalgainrange.has_value());
}

// ── All fields explicitly set ─────────────────────────────────────────────────

TEST(CameraISPConfigTest, AllFieldsOverridden)
{
    std::string path = write_temp_toml(R"(
[camera-isp]
wbmode               = 5
aelock               = true
awblock              = true
ee_mode              = 2
ee_strength          = 0.5
aeantibanding        = 3
exposurecompensation = 1.0
tnr_mode             = 2
tnr_strength         = 0.8
saturation           = 1.5
fps                  = 20
max_buffers          = 4
exposuretimerange    = [13000, 683709000]
gainrange            = [1.0, 16.0]
ispdigitalgainrange  = [1.0, 8.0]
)");

    Configuration cfg;
    cfg.LoadConfiguration(path);
    const CameraISPConfig& isp = cfg.GetCameraISPConfig();

    EXPECT_EQ(isp.wbmode, 5);
    EXPECT_EQ(isp.aelock, true);
    EXPECT_EQ(isp.awblock, true);
    EXPECT_EQ(isp.ee_mode, 2);
    EXPECT_FLOAT_EQ(isp.ee_strength, 0.5f);
    EXPECT_EQ(isp.aeantibanding, 3);
    EXPECT_FLOAT_EQ(isp.exposurecompensation, 1.0f);
    EXPECT_EQ(isp.tnr_mode, 2);
    EXPECT_FLOAT_EQ(isp.tnr_strength, 0.8f);
    EXPECT_FLOAT_EQ(isp.saturation, 1.5f);
    EXPECT_EQ(isp.fps, 20);
    EXPECT_EQ(isp.max_buffers, 4);

    ASSERT_TRUE(isp.exposuretimerange.has_value());
    EXPECT_EQ(isp.exposuretimerange->first,  13000);
    EXPECT_EQ(isp.exposuretimerange->second, 683709000);

    ASSERT_TRUE(isp.gainrange.has_value());
    EXPECT_FLOAT_EQ(isp.gainrange->first,  1.0f);
    EXPECT_FLOAT_EQ(isp.gainrange->second, 16.0f);

    ASSERT_TRUE(isp.ispdigitalgainrange.has_value());
    EXPECT_FLOAT_EQ(isp.ispdigitalgainrange->first,  1.0f);
    EXPECT_FLOAT_EQ(isp.ispdigitalgainrange->second, 8.0f);
}

// ── Partial config: only some fields set, rest keep struct defaults ────────────

TEST(CameraISPConfigTest, PartialISPSectionKeepsUnsetDefaults)
{
    std::string path = write_temp_toml(R"(
[camera-isp]
wbmode  = 1
aelock  = true
saturation = 0.5
)");

    Configuration cfg;
    cfg.LoadConfiguration(path);
    const CameraISPConfig& isp = cfg.GetCameraISPConfig();

    // Fields that were set
    EXPECT_EQ(isp.wbmode, 1);
    EXPECT_EQ(isp.aelock, true);
    EXPECT_FLOAT_EQ(isp.saturation, 0.5f);

    // Fields that were omitted — must equal struct defaults
    EXPECT_EQ(isp.awblock, false);
    EXPECT_EQ(isp.ee_mode, 1);
    EXPECT_FLOAT_EQ(isp.ee_strength, -1.0f);
    EXPECT_EQ(isp.aeantibanding, 1);
    EXPECT_FLOAT_EQ(isp.exposurecompensation, 0.0f);
    EXPECT_EQ(isp.tnr_mode, 1);
    EXPECT_FLOAT_EQ(isp.tnr_strength, -1.0f);
    EXPECT_EQ(isp.fps, DEFAULT_CAMERA_FPS);
    EXPECT_EQ(isp.max_buffers, 2);
    EXPECT_FALSE(isp.exposuretimerange.has_value());
    EXPECT_FALSE(isp.gainrange.has_value());
    EXPECT_FALSE(isp.ispdigitalgainrange.has_value());
}

// ── Sentinel values: ee_strength and tnr_strength of -1 mean driver default ───

TEST(CameraISPConfigTest, StrengthSentinelNegativeOnePreserved)
{
    std::string path = write_temp_toml(R"(
[camera-isp]
ee_strength  = -1.0
tnr_strength = -1.0
)");

    Configuration cfg;
    cfg.LoadConfiguration(path);
    const CameraISPConfig& isp = cfg.GetCameraISPConfig();

    EXPECT_FLOAT_EQ(isp.ee_strength,  -1.0f);
    EXPECT_FLOAT_EQ(isp.tnr_strength, -1.0f);
}

TEST(CameraISPConfigTest, StrengthPositiveValueOverridesSentinel)
{
    std::string path = write_temp_toml(R"(
[camera-isp]
ee_strength  = 0.3
tnr_strength = 0.7
)");

    Configuration cfg;
    cfg.LoadConfiguration(path);
    const CameraISPConfig& isp = cfg.GetCameraISPConfig();

    EXPECT_FLOAT_EQ(isp.ee_strength,  0.3f);
    EXPECT_FLOAT_EQ(isp.tnr_strength, 0.7f);
}

// ── wbmode boundary values ────────────────────────────────────────────────────

TEST(CameraISPConfigTest, WbmodeOff)
{
    std::string path = write_temp_toml("[camera-isp]\nwbmode = 0\n");
    Configuration cfg;
    cfg.LoadConfiguration(path);
    EXPECT_EQ(cfg.GetCameraISPConfig().wbmode, 0);
}

TEST(CameraISPConfigTest, WbmodeAuto)
{
    std::string path = write_temp_toml("[camera-isp]\nwbmode = 1\n");
    Configuration cfg;
    cfg.LoadConfiguration(path);
    EXPECT_EQ(cfg.GetCameraISPConfig().wbmode, 1);
}

TEST(CameraISPConfigTest, WbmodeManual)
{
    std::string path = write_temp_toml("[camera-isp]\nwbmode = 9\n");
    Configuration cfg;
    cfg.LoadConfiguration(path);
    EXPECT_EQ(cfg.GetCameraISPConfig().wbmode, 9);
}

// ── Optional ranges absent → std::optional remains empty ─────────────────────

TEST(CameraISPConfigTest, OptionalRangesAbsentWhenNotInTOML)
{
    std::string path = write_temp_toml("[camera-isp]\nwbmode = 0\n");
    Configuration cfg;
    cfg.LoadConfiguration(path);
    const CameraISPConfig& isp = cfg.GetCameraISPConfig();

    EXPECT_FALSE(isp.exposuretimerange.has_value());
    EXPECT_FALSE(isp.gainrange.has_value());
    EXPECT_FALSE(isp.ispdigitalgainrange.has_value());
}

// ── CameraManager: CountConfiguredCameras / enabled-awareness ────────────────

static std::array<CameraConfig, NUM_CAMERAS> make_cam_configs(std::initializer_list<bool> enabled_flags)
{
    std::array<CameraConfig, NUM_CAMERAS> configs;
    size_t i = 0;
    for (bool en : enabled_flags)
    {
        configs[i] = {static_cast<int64_t>(i), "/dev/video" + std::to_string(i), 640, 480, en};
        ++i;
    }
    for (; i < NUM_CAMERAS; ++i)
        configs[i] = {static_cast<int64_t>(i), "/dev/video" + std::to_string(i), 640, 480, false};
    return configs;
}

// ── CountConfiguredCameras ────────────────────────────────────────────────────

TEST(CameraManagerTest, CountConfiguredCameras_AllEnabled)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({true, true, true, true}), CameraISPConfig{}, im);
    EXPECT_EQ(cm.CountConfiguredCameras(), 4);
}

TEST(CameraManagerTest, CountConfiguredCameras_NoneEnabled)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({false, false, false, false}), CameraISPConfig{}, im);
    EXPECT_EQ(cm.CountConfiguredCameras(), 0);
}

TEST(CameraManagerTest, CountConfiguredCameras_PartiallyEnabled)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({true, false, true, false}), CameraISPConfig{}, im);
    EXPECT_EQ(cm.CountConfiguredCameras(), 2);
}

TEST(CameraManagerTest, CountConfiguredCameras_OneEnabled)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({false, false, false, true}), CameraISPConfig{}, im);
    EXPECT_EQ(cm.CountConfiguredCameras(), 1);
}

// ── CountActiveCameras initial state ─────────────────────────────────────────

TEST(CameraManagerTest, CountActiveCameras_ZeroOnConstruction)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({true, true, true, true}), CameraISPConfig{}, im);
    EXPECT_EQ(cm.CountActiveCameras(), 0);
}

// ── EnableCameras: config-disabled cameras are never attempted ────────────────

TEST(CameraManagerTest, EnableCameras_NoneEnabled_ReturnsZeroWithoutAttempt)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({false, false, false, false}), CameraISPConfig{}, im);
    EXPECT_NO_THROW({
        int n = cm.EnableCameras();
        EXPECT_EQ(n, 0);
    });
    EXPECT_EQ(cm.CountActiveCameras(), 0);
}

TEST(CameraManagerTest, EnableCameras_NoHardware_ReturnsZeroActive)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({true, true, true, true}), CameraISPConfig{}, im);
    EXPECT_NO_THROW({
        int n = cm.EnableCameras();
        // Whether hardware is present or not, returned count must match CountActiveCameras().
        EXPECT_EQ(n, cm.CountActiveCameras());
    });
}

// ── DisableCameras: config-disabled cameras are skipped ──────────────────────

TEST(CameraManagerTest, DisableCameras_NoneEnabled_ReturnsZero)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({false, false, false, false}), CameraISPConfig{}, im);
    int n = cm.DisableCameras();
    EXPECT_EQ(n, 0);
}

TEST(CameraManagerTest, DisableCameras_OnlyActsOnConfigEnabledCameras)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({true, false, true, false}), CameraISPConfig{}, im);
    int n = cm.DisableCameras();
    EXPECT_EQ(n, 2);
}

TEST(CameraManagerTest, DisableCameras_AllConfigEnabled_ProcessesAll)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({true, true, true, true}), CameraISPConfig{}, im);
    int n = cm.DisableCameras();
    EXPECT_EQ(n, 4);
}

// ── PrepareForCapture: uses CountConfiguredCameras, not NUM_CAMERAS ───────────

TEST(CameraManagerTest, PrepareForCapture_ReturnsTrueWhenNoCamerasConfigured)
{
    // active(0) == configured(0) → already "fully ready", no EnableCameras call.
    InferenceManager im;
    CameraManager cm(make_cam_configs({false, false, false, false}), CameraISPConfig{}, im);
    EXPECT_TRUE(cm.PrepareForCapture());
    EXPECT_EQ(cm.CountActiveCameras(), 0);
}

TEST(CameraManagerTest, PrepareForCapture_ActiveEqualsConfiguredOnSuccess)
{
    // When PrepareForCapture succeeds, active cameras must equal configured cameras.
    // This validates that it uses CountConfiguredCameras(), not NUM_CAMERAS.
    InferenceManager im;
    CameraManager cm(make_cam_configs({true, true, true, true}), CameraISPConfig{}, im);
    bool ok = cm.PrepareForCapture();
    if (ok)
        EXPECT_EQ(cm.CountActiveCameras(), cm.CountConfiguredCameras());
}

TEST(CameraManagerTest, PrepareForCapture_PartialConfig_ActiveEqualsConfiguredOnSuccess)
{
    // Only 2 cameras configured. If PrepareForCapture succeeds, only 2 should be
    // active — not 4. This is the core regression from the old NUM_CAMERAS comparison.
    InferenceManager im;
    CameraManager cm(make_cam_configs({true, false, true, false}), CameraISPConfig{}, im);
    bool ok = cm.PrepareForCapture();
    if (ok)
        EXPECT_EQ(cm.CountActiveCameras(), cm.CountConfiguredCameras()); // 2, not 4
}
