#include <gtest/gtest.h>
#include "spdlog/spdlog.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

// Expose private members intentionally so we can verify setter side-effects
// and directly unit-test RCPreprocessImg / LDPreprocessImg.
#define private public
#include <inference/inference_manager.hpp>
#include <inference/runtimes.hpp>  // full LDNet/RCNet definitions for member access
#undef private

#include <NvInfer.h>
#include <opencv2/opencv.hpp>

// ============================================================
// Helpers
// ============================================================

static std::shared_ptr<Frame> MakeSyntheticFrame(int w = 640, int h = 480)
{
    return std::make_shared<Frame>(0, cv::Mat(h, w, CV_8UC3, cv::Scalar(100, 150, 200)), 0ULL);
}

static nvinfer1::Dims MakeDims(std::initializer_list<int64_t> sizes)
{
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int>(sizes.size());
    int i = 0;
    for (int64_t s : sizes) dims.d[i++] = s;
    return dims;
}

// ============================================================
// Fixture shared by all InferenceManager tests
// ============================================================

class OrchestratorTest : public ::testing::Test {
protected:
    InferenceManager im;
};

// ============================================================
// Parameterized: LDNetConfig::GetFileNameAppendix
// ============================================================

struct LDNetConfigCase {
    NET_QUANTIZATION quant;
    int width;
    bool embedded_nms;
    bool use_trt;
    std::string_view expected;
};

class LDNetConfigTest : public ::testing::TestWithParam<LDNetConfigCase> {};

TEST_P(LDNetConfigTest, GetFileNameAppendix)
{
    auto& p = GetParam();
    LDNetConfig cfg{p.quant, p.width, p.width, p.embedded_nms, p.use_trt};
    EXPECT_EQ(cfg.GetFileNameAppendix(), p.expected);
}

INSTANTIATE_TEST_SUITE_P(NNetConfiguration, LDNetConfigTest, ::testing::Values(
    LDNetConfigCase{NET_QUANTIZATION::FP16, 4608, false, true,  "_weights_fp16_sz_4608.trt"},
    LDNetConfigCase{NET_QUANTIZATION::FP32, 4608, false, false, "_weights_fp32_sz_4608.onnx"},
    LDNetConfigCase{NET_QUANTIZATION::INT8, 4608, true,  true,  "_weights_int8_sz_4608_nms.trt"},
    LDNetConfigCase{NET_QUANTIZATION::FP16, 4608, true,  false, "_weights_fp16_sz_4608_nms.onnx"},
    LDNetConfigCase{NET_QUANTIZATION::FP32, 1024, false, false, "_weights_fp32_sz_1024.onnx"}
));

// ============================================================
// Parameterized: Orchestrator::GetMemorySize
// ============================================================

struct MemorySizeCase {
    std::vector<int64_t> dims_vec;
    size_t element_size;
    size_t expected;
};

class GetMemorySizeTest : public ::testing::TestWithParam<MemorySizeCase> {};

TEST_P(GetMemorySizeTest, ComputesCorrectly)
{
    auto& p = GetParam();
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int>(p.dims_vec.size());
    for (int i = 0; i < dims.nbDims; i++) dims.d[i] = p.dims_vec[i];
    EXPECT_EQ(InferenceManager::GetMemorySize(dims, p.element_size), p.expected);
}

INSTANTIATE_TEST_SUITE_P(NNetConfiguration, GetMemorySizeTest, ::testing::Values(
    MemorySizeCase{{10},               sizeof(float), 10 * sizeof(float)},
    MemorySizeCase{{2, 3, 4},          sizeof(float), 24 * sizeof(float)},
    MemorySizeCase{{},                 sizeof(float), sizeof(float)},          // 0 dims → just element_size
    MemorySizeCase{{1, 3, 2592, 4608}, 1,             1ULL * 3 * 2592 * 4608},
    MemorySizeCase{{100},              2,              200UL},
    MemorySizeCase{{100},              8,              800UL}
));

// Dynamic TensorRT dim (-1) must return 0, not wrap around as a giant size_t
TEST(NNetConfiguration, GetMemorySize_DynamicDim_ReturnsZero)
{
    nvinfer1::Dims dims = MakeDims({1, 3, -1, -1}); // e.g. dynamic H×W
    EXPECT_EQ(InferenceManager::GetMemorySize(dims, sizeof(float)), 0UL);
}

// ============================================================
// Setters — bool flags
// ============================================================

TEST_F(OrchestratorTest, SetPreloadRCEngine_UpdatesFlag)
{
    im.SetPreloadRCEngine(true);  EXPECT_TRUE(im.preload_rc_engine_);
    im.SetPreloadRCEngine(false); EXPECT_FALSE(im.preload_rc_engine_);
}

TEST_F(OrchestratorTest, SetPreloadLDEngines_UpdatesFlag)
{
    im.SetPreloadLDEngines(true);  EXPECT_TRUE(im.preload_ld_engines_);
    im.SetPreloadLDEngines(false); EXPECT_FALSE(im.preload_ld_engines_);
}

// ============================================================
// Setters — path validation (bad path → silent reject)
// ============================================================

TEST_F(OrchestratorTest, SetRCNetEnginePath_InvalidPath_Rejected)
{
    std::string original = im.rc_engine_path_;
    EXPECT_EQ(im.SetRCNetEnginePath("/bad/path/engine.trt"), EC::FILE_DOES_NOT_EXIST);
    EXPECT_EQ(im.rc_engine_path_, original);
}

TEST_F(OrchestratorTest, SetRCNetEnginePath_EmptyPath_Rejected)
{
    std::string original = im.rc_engine_path_;
    EXPECT_EQ(im.SetRCNetEnginePath(""), EC::FILE_DOES_NOT_EXIST);
    EXPECT_EQ(im.rc_engine_path_, original);
}

TEST_F(OrchestratorTest, SetLDNetEngineFolderPath_InvalidPath_Rejected)
{
    std::string original = im.ld_engine_folder_path_;
    EXPECT_EQ(im.SetLDNetEngineFolderPath("/bad/folder"), EC::FILE_DOES_NOT_EXIST);
    EXPECT_EQ(im.ld_engine_folder_path_, original);
}

// ============================================================
// SetLDNetConfig
// ============================================================

TEST_F(OrchestratorTest, SetLDNetConfig_AllFieldsUpdated)
{
    im.SetLDNetConfig(NET_QUANTIZATION::INT8, 1024, 768, true, false);
    EXPECT_EQ(im.ldnet_config_.weight_quant, NET_QUANTIZATION::INT8);
    EXPECT_EQ(im.ldnet_config_.input_width,  1024);
    EXPECT_EQ(im.ldnet_config_.input_height, 768);
    EXPECT_TRUE(im.ldnet_config_.embedded_nms);
    EXPECT_FALSE(im.ldnet_config_.use_trt);
    // Cross-check via appendix so the struct is actually being used correctly
    EXPECT_EQ(im.ldnet_config_.GetFileNameAppendix(), "_weights_int8_sz_1024_nms.onnx");
}

TEST_F(OrchestratorTest, SetLDNetConfig_CanBeOverwritten)
{
    im.SetLDNetConfig(NET_QUANTIZATION::FP16, 4608, 2592, false, true);
    im.SetLDNetConfig(NET_QUANTIZATION::FP32, 2048, 2048, true,  false);
    EXPECT_EQ(im.ldnet_config_.weight_quant, NET_QUANTIZATION::FP32);
    EXPECT_EQ(im.ldnet_config_.input_width,  2048);
}

// ============================================================
// GrabNewImage
// ============================================================

TEST_F(OrchestratorTest, GrabNewImage_NullFrame_IsNoOp)
{
    EXPECT_NO_FATAL_FAILURE(im.GrabNewImage(nullptr));
    EXPECT_EQ(im.current_frame_, nullptr);
}

TEST_F(OrchestratorTest, GrabNewImage_ValidFrame_SetsPointerAndResetsCounters)
{
    im.num_rc_inferences_on_current_frame_ = 5;
    im.num_ld_inferences_on_current_frame_ = 3;
    auto frame = MakeSyntheticFrame();
    im.GrabNewImage(frame);

    EXPECT_EQ(im.current_frame_, frame);
    EXPECT_EQ(im.num_rc_inferences_on_current_frame_, 0);
    EXPECT_EQ(im.num_ld_inferences_on_current_frame_, 0);
}

TEST_F(OrchestratorTest, GrabNewImage_SuccessiveCallsReplaceFrame)
{
    im.GrabNewImage(MakeSyntheticFrame(320, 240));
    auto frame2 = MakeSyntheticFrame(640, 480);
    im.GrabNewImage(frame2);
    EXPECT_EQ(im.current_frame_, frame2);
}

// ============================================================
// ExecRCInference — error paths
// ============================================================

TEST_F(OrchestratorTest, ExecRCInference_NoFrame_ReturnsNoFrameError)
{
    EXPECT_EQ(im.ExecRCInference(), EC::NN_NO_FRAME_AVAILABLE);
}

// Both preload modes should end in NN_FAILED_TO_OPEN_ENGINE_FILE when the engine file is absent.
// ExecRCInference always attempts a load if the engine is uninitialized, regardless of the
// preload flag. The constructor may load the real engine if default paths exist, so we explicitly
// unload it and redirect rc_engine_path_ to a non-existent file via the private member.
TEST_F(OrchestratorTest, ExecRCInference_EngineAbsent_PreloadOn_ReturnsFailedToOpenEngine)
{
    im.FreeRCNet();
    im.rc_engine_path_ = "/nonexistent/path.trt"; // bypass filesystem-validated setter
    im.SetPreloadRCEngine(true);
    im.GrabNewImage(MakeSyntheticFrame());
    // preload=true but engine absent → load attempt fails to open file
    EXPECT_EQ(im.ExecRCInference(), EC::NN_FAILED_TO_OPEN_ENGINE_FILE);
}

TEST_F(OrchestratorTest, ExecRCInference_EngineAbsent_PreloadOff_ReturnsFailedToOpenEngine)
{
    im.FreeRCNet();
    im.rc_engine_path_ = "/nonexistent/path.trt"; // bypass filesystem-validated setter
    im.SetPreloadRCEngine(false);
    im.GrabNewImage(MakeSyntheticFrame());
    // preload=off and engine absent → load attempt fails to open file
    EXPECT_EQ(im.ExecRCInference(), EC::NN_FAILED_TO_OPEN_ENGINE_FILE);
}

// ============================================================
// ExecLDInference — error paths
// ============================================================

TEST_F(OrchestratorTest, ExecLDInference_NoFrame_ReturnsNoFrameError)
{
    EXPECT_EQ(im.ExecLDInference(), EC::NN_NO_FRAME_AVAILABLE);
}

TEST_F(OrchestratorTest, ExecLDInference_FrameWithNoRegions_ReturnsOK)
{
    // Frame has no RC detections → LD short-circuits cleanly
    im.GrabNewImage(MakeSyntheticFrame());
    EXPECT_EQ(im.ExecLDInference(), EC::OK);
}

TEST_F(OrchestratorTest, ExecFullInference_NoFrame_ReturnsNoFrameError)
{
    EXPECT_EQ(im.ExecFullInference(), EC::NN_NO_FRAME_AVAILABLE);
}

// ============================================================
// Free methods — must not crash regardless of state
// ============================================================

TEST_F(OrchestratorTest, FreeAll_DoesNotCrash)
{
    EXPECT_NO_FATAL_FAILURE(im.FreeEngines());
    EXPECT_NO_FATAL_FAILURE(im.FreeRCNet());
    EXPECT_NO_FATAL_FAILURE(im.FreeLDNets());
    EXPECT_NO_FATAL_FAILURE(im.FreeEngines()); // double-free safe
}

TEST_F(OrchestratorTest, FreeLDNetForRegion_RegionNotInMap_DoesNotCrash)
{
    // ld_nets_ is empty (no model folder found) — must be a silent no-op
    EXPECT_NO_FATAL_FAILURE(im.FreeLDNetForRegion(RegionID::R_17T));
}

// ============================================================
// Private: RCPreprocessImg
// ============================================================

TEST_F(OrchestratorTest, RCPreprocessImg_EmptyImage_OutputReleased)
{
    cv::Mat output(10, 10, CV_32F, cv::Scalar(1.0f));
    im.RCPreprocessImg(cv::Mat{}, output);
    EXPECT_TRUE(output.empty());
}

TEST_F(OrchestratorTest, RCPreprocessImg_ValidBGR_ProducesCorrectBlob)
{
    cv::Mat output;
    im.RCPreprocessImg(cv::Mat(480, 640, CV_8UC3, cv::Scalar(100, 150, 200)), output);
    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.depth(), CV_32F);
    EXPECT_EQ(output.total(), 1UL * 3 * 224 * 224); // NCHW blob
}

TEST_F(OrchestratorTest, RCPreprocessImg_ValuesAreImageNetNormalised)
{
    cv::Mat output;
    im.RCPreprocessImg(cv::Mat(224, 224, CV_8UC3, cv::Scalar(128, 128, 128)), output);
    // blobFromImageWithParams produces a 4D array (1×3×H×W); minMaxLoc only accepts ≤2D.
    // Use minMaxIdx which works on N-dimensional arrays.
    double min_val, max_val;
    cv::minMaxIdx(output, &min_val, &max_val);
    // Mid-grey after ImageNet normalisation is non-zero
    EXPECT_NE(min_val, 0.0);
}

// Parameterized: any source resolution must produce a 1×3×224×224 blob
class RCPreprocessSrcSizeTest : public OrchestratorTest,
                                public ::testing::WithParamInterface<std::pair<int,int>> {};

TEST_P(RCPreprocessSrcSizeTest, AlwaysOutputs224x224)
{
    auto [w, h] = GetParam();
    cv::Mat output;
    im.RCPreprocessImg(cv::Mat(h, w, CV_8UC3, cv::Scalar(50, 100, 200)), output);
    EXPECT_EQ(output.total(), 1UL * 3 * 224 * 224) << "Source: " << w << "x" << h;
}

INSTANTIATE_TEST_SUITE_P(NNetConfiguration, RCPreprocessSrcSizeTest, ::testing::Values(
    std::make_pair(64,   64),
    std::make_pair(1920, 1080),
    std::make_pair(4608, 2592)
));

// ============================================================
// Private: LDPreprocessImg
// ============================================================

TEST_F(OrchestratorTest, LDPreprocessImg_ValuesNormalisedTo0_1)
{
    cv::Mat output;
    im.LDPreprocessImg(cv::Mat(256, 256, CV_8UC3, cv::Scalar(255, 255, 255)), output, 256, 256);
    // Same issue as RC: blob is 4D, use minMaxIdx instead of minMaxLoc.
    double min_val, max_val;
    cv::minMaxIdx(output, &min_val, &max_val);
    EXPECT_GE(min_val, 0.0);
    EXPECT_LE(max_val, 1.0 + 1e-5);
}

// Parameterized: letterboxing always produces exactly the requested target shape
struct LDPreprocCase { int src_w, src_h, tgt_w, tgt_h; };

class LDPreprocessTargetTest : public OrchestratorTest,
                               public ::testing::WithParamInterface<LDPreprocCase> {};

TEST_P(LDPreprocessTargetTest, OutputMatchesTargetDimensions)
{
    auto& p = GetParam();
    cv::Mat output;
    im.LDPreprocessImg(cv::Mat(p.src_h, p.src_w, CV_8UC3, cv::Scalar(50, 100, 150)), output, p.tgt_w, p.tgt_h);
    EXPECT_EQ(output.depth(), CV_32F);
    EXPECT_EQ(output.total(), static_cast<size_t>(1 * 3 * p.tgt_h * p.tgt_w));
}

INSTANTIATE_TEST_SUITE_P(NNetConfiguration, LDPreprocessTargetTest, ::testing::Values(
    LDPreprocCase{640, 480, 512, 256},  // landscape src, non-square target
    LDPreprocCase{300, 400, 128, 128},  // portrait src, square target
    LDPreprocCase{256, 256, 256, 256}   // identical src and target
));

// ============================================================
// GPU memory leak: SetLDNetConfig re-initialization
//
// Strategy: preload all LD engines so GPU memory is allocated,
// then call SetLDNetConfig (which triggers FreeLDNets + reinit).
// cudaMemGetInfo must show the same free memory before and after
// within a small tolerance (CUDA may retain a pool of its own).
// ============================================================

#include <cuda_runtime_api.h>
/**/
TEST_F(OrchestratorTest, SetLDNetConfig_Reinit_NoGPUMemoryLeak)
{
    // Guard: skip if GPU headroom is too low to safely load even one engine.
    // Two engine loads happen during the test (load + reinit), so require enough
    // margin to avoid OOM-crashing the Jetson.
    constexpr size_t kMinFreeBytes = 512ULL * 1024 * 1024; // 512 MiB
    size_t free_initial, total;
    cudaMemGetInfo(&free_initial, &total);
    if (free_initial < kMinFreeBytes)
    {
        GTEST_SKIP() << "Insufficient GPU memory (" << (free_initial >> 20)
                     << " MiB free, need " << (kMinFreeBytes >> 20) << " MiB). Skipping.";
    }

    // Load a single region engine — sufficient to detect a leak without
    // risking OOM from loading the full fleet.
    if (im.ld_nets_.empty())
    {
        GTEST_SKIP() << "No LDNet runtimes initialised (missing model assets). Skipping.";
    }
    RegionID test_region = im.ld_nets_.begin()->first;
    ASSERT_EQ(im.LoadLDNetEngineForRegion(test_region), EC::OK);

    size_t free_before, free_after;
    cudaMemGetInfo(&free_before, &total);

    // Changing config must free the loaded engine before rebuilding.
    im.SetLDNetConfig(im.ldnet_config_.weight_quant,
                       im.ldnet_config_.input_width,
                       im.ldnet_config_.input_height,
                       im.ldnet_config_.embedded_nms,
                       im.ldnet_config_.use_trt);

    cudaMemGetInfo(&free_after, &total);

    // Allow up to 8 MiB drift for CUDA's internal caching
    constexpr size_t tolerance = 8ULL * 1024 * 1024;
    EXPECT_NEAR(static_cast<double>(free_after),
                static_cast<double>(free_before),
                static_cast<double>(tolerance))
        << "GPU memory before: " << free_before / (1024*1024) << " MiB, "
        << "after: "             << free_after  / (1024*1024) << " MiB";
}

// ============================================================
// OOM guard tests — no real GPU memory is consumed.
//
// Each test injects an impossibly large reserve value so the
// pre-flight check fires before any CUDA allocation is made.
// SIZE_MAX/2 is used as the reserve to avoid size_t overflow
// in the guard's overflow-safe comparison.
// ============================================================

TEST_F(OrchestratorTest, LoadEngine_InsufficientGPUMemory_ReturnsError)
{
    if (im.ld_nets_.empty())
        GTEST_SKIP() << "No LDNet runtimes available (missing model assets).";

    RegionID region = im.ld_nets_.begin()->first;
    // Force the guard to fire: require SIZE_MAX/2 bytes beyond the 2.5x estimate.
    // The check is overflow-safe, so this always exceeds real free memory.
    im.ld_nets_[region]->gpu_reserve_bytes_ = SIZE_MAX / 2;

    EXPECT_EQ(im.LoadLDNetEngineForRegion(region), EC::NN_INSUFFICIENT_GPU_MEMORY);
    EXPECT_FALSE(im.ld_nets_[region]->IsInitialized()); // no engine was loaded

    im.ld_nets_[region]->gpu_reserve_bytes_ = 0; // restore
}

TEST_F(OrchestratorTest, EnsureScratchBuffers_InsufficientGPUMemory_ReturnsError)
{
    // Needs one real engine load to get past the initialized/context check.
    constexpr size_t kMinFree = 128ULL * 1024 * 1024;
    size_t gpu_free, gpu_total;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    if (gpu_free < kMinFree)
        GTEST_SKIP() << "Only " << (gpu_free >> 20) << " MiB GPU free — need 128 MiB to load one engine.";
    if (im.ld_nets_.empty())
        GTEST_SKIP() << "No LDNet runtimes available (missing model assets).";

    RegionID region = im.ld_nets_.begin()->first;
    ASSERT_EQ(im.LoadLDNetEngineForRegion(region), EC::OK);

    im.ld_nets_[region]->gpu_reserve_bytes_ = SIZE_MAX / 2;
    EXPECT_EQ(im.ld_nets_[region]->EnsureScratchBuffers(), EC::NN_INSUFFICIENT_GPU_MEMORY);

    im.ld_nets_[region]->gpu_reserve_bytes_ = 0;
    im.FreeLDNetForRegion(region);
}

TEST_F(OrchestratorTest, LoadLDNetEngines_LowMemory_StopsBeforeFirstLoad)
{
    // Drive the between-load threshold above any realistic free memory so
    // LoadLDNetEngines() halts immediately without touching the GPU.
    im.min_gpu_free_between_loads_ = SIZE_MAX / 2;
    im.SetPreloadLDEngines(true);
    im.LoadLDNetEngines();

    for (const auto& [region_id, ld_net] : im.ld_nets_)
    {
        EXPECT_FALSE(ld_net->IsInitialized())
            << "Region " << GetRegionString(region_id) << " should not be loaded under simulated low GPU memory.";
    }

    im.min_gpu_free_between_loads_ = 256ULL * 1024 * 1024; // restore
}

// ============================================================
// Integration tests — Full inference on real sample images
//
//
// Asserts:
//   1. RC net detects the expected region.
//   2. At least one landmark is a true positive (IoU > 0.5
//      with a ground-truth box of the same class_id).
//
// Skipped (not failed) when: TRT engine missing, sample image
// missing, or GPU free memory < 768 MiB.
// ============================================================

namespace {
namespace fs = std::filesystem;

struct SampleImageCase {
    std::string region_str;  // e.g. "17R"
    std::string image_path;  // absolute path to .png
    std::string label_path;  // absolute path to .txt
};

struct RegionTestResult {
    enum class Status { PENDING, PASS, FAIL, SKIP } status = Status::PENDING;

    std::string skip_reason;
    std::string error;

    // RC
    std::vector<std::string> detected_regions;
    bool rc_correct = false;

    // Ground truth
    int gt_box_count = 0;

    // LD predictions
    int landmark_count = 0;
    int true_positives = 0;

    struct LandmarkMatch {
        RegionID region_id;
        int   class_id;
        float confidence;
        float best_iou;  // highest IoU against any same-class GT box
        bool  is_tp;     // best_iou > 0.5
    };
    std::vector<LandmarkMatch> matches;
};

static void WriteInferenceReport(const std::map<std::string, RegionTestResult>& results)
{
    const char* env = std::getenv("INFERENCE_REPORT_PATH");
    const std::string path = env ? env : "tests/inference_test_report.md";

    std::ofstream f(path);
    if (!f.is_open()) {
        spdlog::error("Could not open report file: {}", path);
        return;
    }

    // Header
    int pass = 0, fail = 0, skip = 0;
    for (const auto& [_, r] : results) {
        if (r.status == RegionTestResult::Status::PASS)       ++pass;
        else if (r.status == RegionTestResult::Status::SKIP)  ++skip;
        else                                                   ++fail;
    }

    f << "# Inference Integration Test Report\n\n";
    f << "**Summary:** " << pass << " passed, " << fail << " failed, " << skip << " skipped "
      << "(" << results.size() << " total)\n\n";
    f << "---\n\n";

    for (const auto& [region, r] : results) {
        const char* icon =
            r.status == RegionTestResult::Status::PASS ? "PASS" :
            r.status == RegionTestResult::Status::SKIP ? "SKIP" : "FAIL";

        f << "## [" << icon << "] Region " << region << "\n\n";

        if (r.status == RegionTestResult::Status::SKIP) {
            f << "**Skipped:** " << r.skip_reason << "\n\n";
            continue;
        }

        if (!r.error.empty())
            f << "**Error:** " << r.error << "\n\n";

        // RC
        f << "### RC Classification\n";
        f << "- Expected: `" << region << "`\n";
        f << "- Detected: ";
        if (r.detected_regions.empty()) {
            f << "_none_\n";
        } else {
            for (size_t i = 0; i < r.detected_regions.size(); ++i) {
                if (i) f << ", ";
                f << "`" << r.detected_regions[i] << "`";
            }
            f << "\n";
        }
        f << "- RC correct: " << (r.rc_correct ? "**yes**" : "**NO**") << "\n\n";

        // LD
        f << "### Landmark Detection\n";
        f << "- Ground-truth boxes: " << r.gt_box_count << "\n";
        f << "- Predicted landmarks: " << r.landmark_count << "\n";
        f << "- True positives (IoU > 0.5): " << r.true_positives << "\n";

        if (!r.matches.empty()) {
            auto sorted = r.matches;
            std::sort(sorted.begin(), sorted.end(), [](const RegionTestResult::LandmarkMatch& a,
                                                       const RegionTestResult::LandmarkMatch& b) {
                if (a.region_id != b.region_id) return static_cast<int>(a.region_id) < static_cast<int>(b.region_id);
                return a.class_id < b.class_id;
            });

            // Pre-format all cells so we can measure column widths
            struct Row { std::string region, cls, conf, iou, tp; };
            std::vector<Row> rows;
            rows.reserve(sorted.size());
            for (const auto& m : sorted) {
                std::ostringstream conf_ss, iou_ss;
                conf_ss << std::fixed << std::setprecision(3) << m.confidence;
                iou_ss  << std::fixed << std::setprecision(3) << m.best_iou;
                rows.push_back({std::string(GetRegionString(m.region_id)),
                                std::to_string(m.class_id),
                                conf_ss.str(), iou_ss.str(),
                                m.is_tp ? "yes" : "no"});
            }

            // Column widths (at least as wide as the header)
            size_t w0 = 9, w1 = 8, w2 = 10, w3 = 8, w4 = 3; // "region_id","class_id","confidence","best_iou","TP?"
            for (const auto& row : rows) {
                w0 = std::max(w0, row.region.size());
                w1 = std::max(w1, row.cls.size());
                w2 = std::max(w2, row.conf.size());
                w3 = std::max(w3, row.iou.size());
                w4 = std::max(w4, row.tp.size());
            }

            auto cell = [](const std::string& s, size_t w) {
                return s + std::string(w - s.size(), ' ');
            };

            f << "\n"
              << "| " << cell("region_id", w0) << " | " << cell("class_id",   w1)
              << " | " << cell("confidence", w2) << " | " << cell("best_iou",  w3)
              << " | " << cell("TP?", w4) << " |\n";
            f << "| " << std::string(w0, '-') << " | " << std::string(w1, '-')
              << " | " << std::string(w2, '-') << " | " << std::string(w3, '-')
              << " | " << std::string(w4, '-') << " |\n";
            for (const auto& row : rows) {
                f << "| " << cell(row.region, w0) << " | " << cell(row.cls, w1)
                  << " | " << cell(row.conf,   w2) << " | " << cell(row.iou, w3)
                  << " | " << cell(row.tp,     w4) << " |\n";
            }
        }
        f << "\n";
    }

    f.flush();
    spdlog::info("Inference test report written to: {}", path);
}

// Collect one labeled image per region from the sample_images folder.
// Expected naming: l8_{regionstr}_{imageid}.jpg or .png with a matching .txt label.
// Called once at link-time via ValuesIn(); returns empty when assets absent.
static std::vector<SampleImageCase> CollectSampleCases()
{
    std::vector<SampleImageCase> result;
    const fs::path sample_dir = fs::path(MODELS_DIR) / "V1" / "sample_images";
    if (!fs::exists(sample_dir)) return result;

    for (RegionID rid : GetAllRegionIDs())
    {
        const std::string region_str = std::string(GetRegionString(rid));
        const fs::path region_dir = sample_dir / region_str;
        if (!fs::exists(region_dir) || !fs::is_directory(region_dir)) continue;
        const std::string expected_prefix = "l8_" + region_str + "_";
        for (const auto& entry : fs::directory_iterator(region_dir))
        {
            if (!entry.is_regular_file()) continue;
            const auto ext = entry.path().extension();
            if (ext != ".jpg" && ext != ".png") continue;
            const std::string fname = entry.path().filename().string();
            if (fname.rfind(expected_prefix, 0) != 0) continue;
            fs::path txt = entry.path();
            txt.replace_extension(".txt");
            if (!fs::exists(txt) || fs::file_size(txt) == 0) continue;
            result.push_back({region_str, entry.path().string(), txt.string()});
            break; // one labeled example per region is sufficient
        }
    }
    return result;
}

static float ComputeIoU(const cv::Rect& a, const cv::Rect& b)
{
    const int inter = (a & b).area();
    const int uni   = (a | b).area();
    return (uni == 0) ? 0.0f : static_cast<float>(inter) / static_cast<float>(uni);
}

} // namespace

class InferenceIntegrationTest : public ::testing::TestWithParam<SampleImageCase>
{
protected:
    // One InferenceManager for the entire suite — avoids repeated TRT context
    static InferenceManager* orc_;
    static std::map<std::string, RegionTestResult> results_;

    static void SetUpTestSuite()
    {
        const std::string models = MODELS_DIR;
        orc_ = new InferenceManager();
        if (orc_->SetRCNetEnginePath(models + "/V1/trained-rc/effnet_0997acc.trt") != EC::OK)
        {
            delete orc_;
            orc_ = nullptr;
            return;
        }
        orc_->SetLDNetEngineFolderPath(models + "/V1/trained-ld");
    }

    static void TearDownTestSuite()
    {
        WriteInferenceReport(results_);
        delete orc_;
        orc_ = nullptr;
    }
};

InferenceManager* InferenceIntegrationTest::orc_ = nullptr;
std::map<std::string, RegionTestResult> InferenceIntegrationTest::results_;

TEST_P(InferenceIntegrationTest, FullInference_DetectsRegionAndTruePositiveLandmarks)
{
    const auto& p = GetParam();
    RegionTestResult& result = results_[p.region_str];
    result.status = RegionTestResult::Status::FAIL; // default until proven otherwise

    if (!orc_) {
        result.status = RegionTestResult::Status::SKIP;
        result.skip_reason = "RC TRT engine not found";
        GTEST_SKIP() << result.skip_reason;
    }

    // Guard: TRT FP16 engine exists for this region
    const std::string engine_path = std::string(MODELS_DIR) + "/V1/trained-ld/"
                                  + p.region_str + "/" + p.region_str
                                  + "_weights_fp16_sz_4608.trt";
    if (!fs::exists(engine_path)) {
        result.status = RegionTestResult::Status::SKIP;
        result.skip_reason = "TRT engine not found: " + engine_path;
        GTEST_SKIP() << result.skip_reason;
    }

    // Load the sample image from disk
    cv::Mat img = cv::imread(p.image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        result.error = "Failed to read image: " + p.image_path;
        ASSERT_FALSE(img.empty()) << result.error;
    }

    // Free any LD engine left over from the previous test case and sync the
    // device so CUDA's allocator reports accurate free-memory figures before
    // the next load.
    orc_->FreeLDNets();
    cudaDeviceSynchronize();

    auto frame_ptr = std::make_shared<Frame>(0, img, 0ULL);

    {
        const EC infer_status = orc_->ProcessFrame(frame_ptr, ProcessingStage::LDNeted);
        if (infer_status == EC::NN_INSUFFICIENT_GPU_MEMORY) {
            result.status = RegionTestResult::Status::SKIP;
            result.skip_reason = "Insufficient GPU memory";
            GTEST_SKIP() << "Insufficient GPU memory for region " << p.region_str;
        }
        if (infer_status != EC::OK) {
            result.error = "ExecFullInference returned error code " + std::to_string(to_uint8(infer_status));
            ASSERT_EQ(infer_status, EC::OK) << result.error;
        }
    }

    // --- RC: record what regions were detected ---
    const RegionID expected_rid = GetRegionID(p.region_str);
    const auto region_ids = frame_ptr->GetRegionIDs();
    for (const auto& rid : region_ids)
        result.detected_regions.push_back(std::string(GetRegionString(rid)));
    result.rc_correct = std::find(region_ids.begin(), region_ids.end(), expected_rid) != region_ids.end();

    EXPECT_TRUE(result.rc_correct)
        << "RC net did not detect region " << p.region_str
        << " (detected " << region_ids.size() << " region(s)).";

    // --- Parse YOLO ground-truth labels (normalized cx cy w h per line) ---
    std::vector<std::pair<int, cv::Rect>> gt_boxes; // {class_id, pixel-space Rect}
    {
        std::ifstream ifs(p.label_path);
        std::string line;
        while (std::getline(ifs, line))
        {
            std::istringstream iss(line);
            int class_id;
            float nx, ny, nw, nh;
            if (!(iss >> class_id >> nx >> ny >> nw >> nh)) continue;
            const int x = static_cast<int>(nx * img.cols - nw * img.cols * 0.5f);
            const int y = static_cast<int>(ny * img.rows - nh * img.rows * 0.5f);
            const int w = static_cast<int>(nw * img.cols);
            const int h = static_cast<int>(nh * img.rows);
            gt_boxes.push_back({class_id, cv::Rect(x, y, w, h)});
        }
    }
    result.gt_box_count = static_cast<int>(gt_boxes.size());
    if (gt_boxes.empty()) {
        result.error = "No ground-truth labels in " + p.label_path;
        ASSERT_FALSE(gt_boxes.empty()) << result.error;
    }

    // --- LD: record per-landmark match quality ---
    const auto& landmarks = frame_ptr->GetLandmarks();
    result.landmark_count = static_cast<int>(landmarks.size());

    EXPECT_FALSE(landmarks.empty())
        << "No landmarks detected for region " << p.region_str;

    for (const auto& lm : landmarks)
    {
        float best_iou = 0.0f;
        const cv::Rect pred(
            static_cast<int>(lm.x - lm.width  * 0.5f),
            static_cast<int>(lm.y - lm.height * 0.5f),
            static_cast<int>(lm.width),
            static_cast<int>(lm.height));

        for (const auto& [gt_class, gt_box] : gt_boxes)
        {
            if (static_cast<int>(lm.class_id) != gt_class) continue;
            best_iou = std::max(best_iou, ComputeIoU(pred, gt_box));
        }

        const bool is_tp = best_iou > 0.5f;
        if (is_tp) ++result.true_positives;
        result.matches.push_back({lm.region_id, static_cast<int>(lm.class_id), lm.confidence, best_iou, is_tp});
    }

    EXPECT_GT(result.true_positives, 0)
        << "No true positive landmarks (IoU > 0.5) detected for region " << p.region_str;

    if (!::testing::Test::HasFailure())
        result.status = RegionTestResult::Status::PASS;
}

INSTANTIATE_TEST_SUITE_P(
    SampleImages,
    InferenceIntegrationTest,
    ::testing::ValuesIn(CollectSampleCases()),
    [](const ::testing::TestParamInfo<SampleImageCase>& info) {
        return info.param.region_str;
    });
