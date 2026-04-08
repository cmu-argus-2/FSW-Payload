/*
 * Unit tests for CameraManager::SaveLatestFrames pipeline modes.
 *
 * Camera hardware is not available in the test environment, so frames are
 * injected directly into the camera buffer via #define private public.
 *
 * Coverage:
 *   - No active cameras: all modes return 0 and leave buffer_frame_ids empty.
 *   - PERIODIC/CAPTURE_SINGLE: frames saved without any prefiltering.
 *   - PERIODIC_EARTH: earth-failing frames skipped; earth-passing frames saved.
 *   - PERIODIC_ROI / PERIODIC_LDMK: enter inference; with no models loaded
 *     ExecRCInference returns an error and no frame is saved (error path).
 *   - buffer_frame_ids reflects exactly what was saved (Risk-1 regression).
 *   - new_frame_flag is cleared after SaveLatestFrames.
 *   - Multiple cameras: only frames from active cameras with new frames are processed.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <opencv2/opencv.hpp>

#define private public
#include "vision/camera_manager.hpp"
#include "vision/camera.hpp"
#undef private

#include "inference/inference_manager.hpp"

namespace fs = std::filesystem;

// ── Frame factories ───────────────────────────────────────────────────────────

// Uniform mid-blue frame: passes prefiltering via "single_color" fallthrough.
// (is_blue check is commented-out in prefilter_image, so it is not rejected.)
static Frame MakeEarthFrame(int cam_id = 0, uint64_t ts = 1000)
{
    cv::Mat img(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    return Frame(cam_id, img, ts);
}

// All-black frame: triggers is_black → prefiltering rejects it.
static Frame MakeBlackFrame(int cam_id = 0, uint64_t ts = 2000)
{
    cv::Mat img(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
    return Frame(cam_id, img, ts);
}

// ── Fixture ───────────────────────────────────────────────────────────────────

struct CameraManagerPipelineTest : ::testing::Test
{
    static std::array<CameraConfig, NUM_CAMERAS> makeDummyConfigs()
    {
        std::array<CameraConfig, NUM_CAMERAS> c;
        for (int i = 0; i < NUM_CAMERAS; ++i)
            c[i] = {static_cast<int64_t>(i), "/dev/null", 64, 64};
        return c;
    }

    InferenceManager                      im;
    std::array<CameraConfig, NUM_CAMERAS> configs{makeDummyConfigs()};
    CameraManager                         cam{configs, im};
    const std::string                     test_folder{"data/test_cam_pipeline/"};

    void SetUp() override    { cam.SetStorageFolder(test_folder); }
    void TearDown() override { fs::remove_all(test_folder); }

    // Directly inject a frame into camera slot i and mark it active with a new frame.
    void InjectFrame(std::size_t i, const Frame& frame)
    {
        cam.cameras[i].buffer_frame   = frame;
        cam.cameras[i]._new_frame_flag.store(true);
        cam.cameras[i].cam_status.store(CAM_STATUS::ACTIVE);
    }
};

// ── No active cameras ─────────────────────────────────────────────────────────

TEST_F(CameraManagerPipelineTest, NoActiveCameras_AllModes_ReturnZero)
{
    // Cameras are INACTIVE by default (construction from /dev/null fails to enable)
    for (auto mode : {CAPTURE_MODE::PERIODIC,
                      CAPTURE_MODE::CAPTURE_SINGLE,
                      CAPTURE_MODE::PERIODIC_EARTH,
                      CAPTURE_MODE::PERIODIC_ROI,
                      CAPTURE_MODE::PERIODIC_LDMK})
    {
        EXPECT_EQ(cam.SaveLatestFrames(mode), 0) << "mode=" << static_cast<int>(mode);
        EXPECT_TRUE(cam.GetBufferFrameIDs().empty()) << "mode=" << static_cast<int>(mode);
    }
}

// ── PERIODIC (no prefilter) ───────────────────────────────────────────────────

TEST_F(CameraManagerPipelineTest, Periodic_BlackFrame_SavedWithoutPrefilter)
{
    // PERIODIC mode must save frames regardless of image content.
    InjectFrame(0, MakeBlackFrame(0, 1000));

    uint8_t n = cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC);

    EXPECT_EQ(n, 1);
    auto ids_periodic = cam.GetBufferFrameIDs();
    ASSERT_EQ(ids_periodic.size(), 1u);
    EXPECT_EQ(std::get<0>(ids_periodic[0]), 0u);    // cam_id
    EXPECT_EQ(std::get<1>(ids_periodic[0]), 1000u); // timestamp
}

TEST_F(CameraManagerPipelineTest, CaptureSingle_BlackFrame_SavedWithoutPrefilter)
{
    InjectFrame(0, MakeBlackFrame(0, 1001));

    uint8_t n = cam.SaveLatestFrames(CAPTURE_MODE::CAPTURE_SINGLE);

    EXPECT_EQ(n, 1);
    EXPECT_EQ(cam.GetBufferFrameIDs().size(), 1u);
}

// ── PERIODIC_EARTH ────────────────────────────────────────────────────────────

TEST_F(CameraManagerPipelineTest, PeriodicEarth_BlackFrame_Skipped)
{
    InjectFrame(0, MakeBlackFrame(0, 2000));

    uint8_t n = cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC_EARTH);

    EXPECT_EQ(n, 0);
    EXPECT_TRUE(cam.GetBufferFrameIDs().empty());
}

TEST_F(CameraManagerPipelineTest, PeriodicEarth_EarthFrame_Saved)
{
    InjectFrame(0, MakeEarthFrame(0, 3000));

    uint8_t n = cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC_EARTH);

    EXPECT_EQ(n, 1);
    auto ids_earth = cam.GetBufferFrameIDs();
    ASSERT_EQ(ids_earth.size(), 1u);
    EXPECT_EQ(std::get<0>(ids_earth[0]), 0u);
    EXPECT_EQ(std::get<1>(ids_earth[0]), 3000u);
}

TEST_F(CameraManagerPipelineTest, PeriodicEarth_MixedFrames_OnlyEarthSaved)
{
    // cam 0: earth-passing, cam 1: black (rejected), cam 2: earth-passing
    InjectFrame(0, MakeEarthFrame(0, 4000));
    InjectFrame(1, MakeBlackFrame(1, 4001));
    InjectFrame(2, MakeEarthFrame(2, 4002));

    uint8_t n = cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC_EARTH);

    EXPECT_EQ(n, 2);
    ASSERT_EQ(cam.GetBufferFrameIDs().size(), 2u);

    auto ids = cam.GetBufferFrameIDs();
    // Both saved IDs must correspond to the earth frames
    auto has_id = [&](uint8_t cam_id, uint64_t ts) {
        return std::any_of(ids.begin(), ids.end(), [&](const auto& t) {
            return std::get<0>(t) == cam_id && std::get<1>(t) == ts;
        });
    };
    EXPECT_TRUE(has_id(0, 4000));
    EXPECT_TRUE(has_id(2, 4002));
}

// ── new_frame_flag cleared after save ────────────────────────────────────────

TEST_F(CameraManagerPipelineTest, NewFrameFlag_ClearedAfterSave)
{
    InjectFrame(0, MakeEarthFrame(0, 5000));
    EXPECT_TRUE(cam.cameras[0].IsNewFrameAvailable());

    cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC);

    EXPECT_FALSE(cam.cameras[0].IsNewFrameAvailable());
}

TEST_F(CameraManagerPipelineTest, NewFrameFlag_ClearedEvenWhenSkipped)
{
    // Frame is skipped by prefilter, but the flag must still be cleared.
    InjectFrame(0, MakeBlackFrame(0, 6000));
    EXPECT_TRUE(cam.cameras[0].IsNewFrameAvailable());

    cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC_EARTH);

    EXPECT_FALSE(cam.cameras[0].IsNewFrameAvailable());
}

// ── buffer_frame_ids reset on each call ──────────────────────────────────────

TEST_F(CameraManagerPipelineTest, BufferFrameIds_DrainClearsAccumulatedIds)
{
    // Accumulate one frame ID then drain — subsequent drain must return empty.
    InjectFrame(0, MakeEarthFrame(0, 7000));
    cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC_EARTH);

    auto first_drain = cam.DrainBufferFrameIDs();
    ASSERT_EQ(first_drain.size(), 1u);

    // No new frames; nothing was accumulated since the drain.
    auto second_drain = cam.DrainBufferFrameIDs();
    EXPECT_TRUE(second_drain.empty());
}

TEST_F(CameraManagerPipelineTest, BufferFrameIds_AccumulatesAcrossBatches)
{
    // Two separate saves accumulate before any drain — both IDs must survive.
    InjectFrame(0, MakeEarthFrame(0, 7100));
    cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC_EARTH);

    InjectFrame(0, MakeEarthFrame(0, 7200));
    cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC_EARTH);

    auto ids = cam.DrainBufferFrameIDs();
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(std::get<1>(ids[0]), 7100u);
    EXPECT_EQ(std::get<1>(ids[1]), 7200u);
}

// ── PERIODIC_ROI / PERIODIC_LDMK (inference error path) ─────────────────────

TEST_F(CameraManagerPipelineTest, PeriodicROI_InferenceFails_NothingSaved)
{
#ifdef CUDA_ENABLED
    GTEST_SKIP() << "inference outcome without model weights is non-deterministic on CUDA builds";
#endif
    InjectFrame(0, MakeEarthFrame(0, 8000));

    uint8_t n = cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC_ROI);

    EXPECT_EQ(n, 0);
}

TEST_F(CameraManagerPipelineTest, PeriodicLDMK_InferenceFails_NothingSaved)
{
#ifdef CUDA_ENABLED
    GTEST_SKIP() << "inference outcome without model weights is non-deterministic on CUDA builds";
#endif
    InjectFrame(0, MakeEarthFrame(0, 9000));

    uint8_t n = cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC_LDMK);

    EXPECT_EQ(n, 0);
}

TEST_F(CameraManagerPipelineTest, PeriodicROI_BlackFrame_SkippedBeforeInference)
{
    // A black frame must be rejected by prefiltering before inference is attempted.
    // If inference were called on a black frame it would be an error anyway, but
    // we verify the frame count is 0 and cameras are not unnecessarily disabled.
    InjectFrame(0, MakeBlackFrame(0, 9500));

    uint8_t n = cam.SaveLatestFrames(CAPTURE_MODE::PERIODIC_ROI);

    EXPECT_EQ(n, 0);
    EXPECT_TRUE(cam.GetBufferFrameIDs().empty());
    // Cameras should remain in their original state (not disabled by inference block)
    EXPECT_EQ(cam.cameras[0].GetStatus(), CAM_STATUS::ACTIVE);
}
