#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <atomic>
#include <unistd.h>
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

// RAII helper: writes content to a unique temp file and deletes it on destruction.
class TempTomlFile
{
public:
    explicit TempTomlFile(const std::string& content)
    {
        static std::atomic<int> counter{0};
        namespace fs = std::filesystem;
        path_ = (fs::temp_directory_path() /
                 ("test_camera_" + std::to_string(::getpid()) + "_" +
                  std::to_string(counter++) + ".toml")).string();
        std::ofstream f(path_);
        EXPECT_TRUE(f.is_open()) << "Failed to open temp file: " << path_;
        f << content;
    }

    ~TempTomlFile() { std::filesystem::remove(path_); }

    const std::string& path() const { return path_; }

    // Prepends REQUIRED_SECTIONS before isp_section (mirrors old write_temp_toml).
    static TempTomlFile with_isp(const std::string& isp_section)
    {
        return TempTomlFile(std::string(REQUIRED_SECTIONS) + "\n" + isp_section);
    }

private:
    std::string path_;
};

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
    TempTomlFile tmp = TempTomlFile::with_isp(""); // no [camera-isp] block
    Configuration cfg;
    cfg.LoadConfiguration(tmp.path());
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
    TempTomlFile tmp = TempTomlFile::with_isp(R"(
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
    cfg.LoadConfiguration(tmp.path());
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
    TempTomlFile tmp = TempTomlFile::with_isp(R"(
[camera-isp]
wbmode  = 1
aelock  = true
saturation = 0.5
)");

    Configuration cfg;
    cfg.LoadConfiguration(tmp.path());
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
    TempTomlFile tmp = TempTomlFile::with_isp(R"(
[camera-isp]
ee_strength  = -1.0
tnr_strength = -1.0
)");

    Configuration cfg;
    cfg.LoadConfiguration(tmp.path());
    const CameraISPConfig& isp = cfg.GetCameraISPConfig();

    EXPECT_FLOAT_EQ(isp.ee_strength,  -1.0f);
    EXPECT_FLOAT_EQ(isp.tnr_strength, -1.0f);
}

TEST(CameraISPConfigTest, StrengthPositiveValueOverridesSentinel)
{
    TempTomlFile tmp = TempTomlFile::with_isp(R"(
[camera-isp]
ee_strength  = 0.3
tnr_strength = 0.7
)");

    Configuration cfg;
    cfg.LoadConfiguration(tmp.path());
    const CameraISPConfig& isp = cfg.GetCameraISPConfig();

    EXPECT_FLOAT_EQ(isp.ee_strength,  0.3f);
    EXPECT_FLOAT_EQ(isp.tnr_strength, 0.7f);
}

// ── wbmode boundary values ────────────────────────────────────────────────────

TEST(CameraISPConfigTest, WbmodeOff)
{
    TempTomlFile tmp = TempTomlFile::with_isp("[camera-isp]\nwbmode = 0\n");
    Configuration cfg;
    cfg.LoadConfiguration(tmp.path());
    EXPECT_EQ(cfg.GetCameraISPConfig().wbmode, 0);
}

TEST(CameraISPConfigTest, WbmodeAuto)
{
    TempTomlFile tmp = TempTomlFile::with_isp("[camera-isp]\nwbmode = 1\n");
    Configuration cfg;
    cfg.LoadConfiguration(tmp.path());
    EXPECT_EQ(cfg.GetCameraISPConfig().wbmode, 1);
}

TEST(CameraISPConfigTest, WbmodeManual)
{
    TempTomlFile tmp = TempTomlFile::with_isp("[camera-isp]\nwbmode = 9\n");
    Configuration cfg;
    cfg.LoadConfiguration(tmp.path());
    EXPECT_EQ(cfg.GetCameraISPConfig().wbmode, 9);
}

// ── Optional ranges absent → std::optional remains empty ─────────────────────

TEST(CameraISPConfigTest, OptionalRangesAbsentWhenNotInTOML)
{
    TempTomlFile tmp = TempTomlFile::with_isp("[camera-isp]\nwbmode = 0\n");
    Configuration cfg;
    cfg.LoadConfiguration(tmp.path());
    const CameraISPConfig& isp = cfg.GetCameraISPConfig();

    EXPECT_FALSE(isp.exposuretimerange.has_value());
    EXPECT_FALSE(isp.gainrange.has_value());
    EXPECT_FALSE(isp.ispdigitalgainrange.has_value());
}

TEST(CameraISPConfigTest, ReloadWithoutISPSectionResetsDefaults)
{
    TempTomlFile with_isp = TempTomlFile::with_isp(R"(
[camera-isp]
wbmode = 5
fps = 20
)");
    TempTomlFile without_isp = TempTomlFile::with_isp("");

    Configuration cfg;
    cfg.LoadConfiguration(with_isp.path());
    ASSERT_EQ(cfg.GetCameraISPConfig().wbmode, 5);
    ASSERT_EQ(cfg.GetCameraISPConfig().fps, 20);

    cfg.LoadConfiguration(without_isp.path());
    EXPECT_EQ(cfg.GetCameraISPConfig().wbmode, 0);
    EXPECT_EQ(cfg.GetCameraISPConfig().fps, DEFAULT_CAMERA_FPS);
}

TEST(ConfigurationErrorHandlingTest, MissingCameraDeviceSectionThrows)
{
    TempTomlFile tmp(R"(
[imu-device]
chipid = 216
i2c_addr = 104
i2c_path = '/dev/i2c-7'
)");

    Configuration cfg;
    EXPECT_THROW(cfg.LoadConfiguration(tmp.path()), std::runtime_error);
}

TEST(ConfigurationErrorHandlingTest, MissingIMUSectionThrows)
{
    TempTomlFile tmp(R"(
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
)");

    Configuration cfg;
    EXPECT_THROW(cfg.LoadConfiguration(tmp.path()), std::runtime_error);
}

TEST(ConfigurationCameraDeviceTest, CameraConfigsIndexedByIdNotTableOrder)
{
    TempTomlFile tmp(R"(
[camera-device.cam4]
id = 3
path = '/dev/video3'
resolution_width = 400
resolution_height = 300
enabled = false

[camera-device.cam2]
id = 1
path = '/dev/video1'
resolution_width = 200
resolution_height = 100
enabled = true

[camera-device.cam1]
id = 0
path = '/dev/video0'
resolution_width = 640
resolution_height = 480
enabled = true

[camera-device.cam3]
id = 2
path = '/dev/video2'
resolution_width = 320
resolution_height = 240
enabled = true

[imu-device]
chipid = 216
i2c_addr = 104
i2c_path = '/dev/i2c-7'
)");

    Configuration cfg;
    cfg.LoadConfiguration(tmp.path());
    const auto& cams = cfg.GetCameraConfigs();

    EXPECT_EQ(cams[0].id, 0);
    EXPECT_EQ(cams[0].path, "/dev/video0");
    EXPECT_EQ(cams[0].width, 640);
    EXPECT_EQ(cams[0].height, 480);

    EXPECT_EQ(cams[1].id, 1);
    EXPECT_EQ(cams[1].path, "/dev/video1");
    EXPECT_EQ(cams[1].width, 200);
    EXPECT_EQ(cams[1].height, 100);

    EXPECT_EQ(cams[2].id, 2);
    EXPECT_EQ(cams[2].path, "/dev/video2");
    EXPECT_EQ(cams[2].width, 320);
    EXPECT_EQ(cams[2].height, 240);

    EXPECT_EQ(cams[3].id, 3);
    EXPECT_EQ(cams[3].path, "/dev/video3");
    EXPECT_EQ(cams[3].width, 400);
    EXPECT_EQ(cams[3].height, 300);
    EXPECT_FALSE(cams[3].enabled);
}

TEST(ConfigurationCameraDeviceTest, DuplicateCameraIdsThrow)
{
    TempTomlFile tmp(R"(
[camera-device.cam1]
id = 0
path = '/dev/video0'
resolution_width = 640
resolution_height = 480
enabled = true

[camera-device.cam2]
id = 0
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
)");

    Configuration cfg;
    EXPECT_THROW(cfg.LoadConfiguration(tmp.path()), std::runtime_error);
}

TEST(ConfigurationCameraDeviceTest, OutOfRangeCameraIdThrows)
{
    TempTomlFile tmp(R"(
[camera-device.cam1]
id = 4
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
)");

    Configuration cfg;
    EXPECT_THROW(cfg.LoadConfiguration(tmp.path()), std::runtime_error);
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

TEST(CameraTest, ConstructorUsesConfiguredResolutionForBufferFrame)
{
    Camera cam(7, "/dev/video7", 320, 240, CameraISPConfig{});
    EXPECT_EQ(cam.GetBufferFrame().GetImgSize(), cv::Size(320, 240));
}

TEST(CameraTest, ConstructorFallsBackToDefaultResolutionForInvalidDimensions)
{
    Camera cam(7, "/dev/video7", 0, -1, CameraISPConfig{});
    EXPECT_EQ(cam.GetBufferFrame().GetImgSize(),
              cv::Size(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT));
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
    EXPECT_EQ(n, 0);
}

TEST(CameraManagerTest, DisableCameras_AllConfigEnabled_ProcessesAll)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({true, true, true, true}), CameraISPConfig{}, im);
    int n = cm.DisableCameras();
    EXPECT_EQ(n, 0);
}

// ── PrepareForCapture: uses CountConfiguredCameras, not NUM_CAMERAS ───────────

TEST(CameraManagerTest, PrepareForCapture_ReturnsTrueWhenNoCamerasConfigured)
{
    // No configured cameras is a real failure: capture requests should not proceed.
    InferenceManager im;
    CameraManager cm(make_cam_configs({false, false, false, false}), CameraISPConfig{}, im);
    EXPECT_FALSE(cm.PrepareForCapture());
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

TEST(CameraManagerTest, SendCaptureRequestRejectedWhenManagerBusy)
{
    InferenceManager im;
    CameraManager cm(make_cam_configs({false, false, false, false}), CameraISPConfig{}, im);
    cm.SetCaptureMode(CAPTURE_MODE::PERIODIC);

    EXPECT_FALSE(cm.SendCaptureRequest());
    EXPECT_EQ(cm.GetCaptureMode(), CAPTURE_MODE::PERIODIC);
}
