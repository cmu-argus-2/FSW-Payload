#include <gtest/gtest.h>
#include "spdlog/spdlog.h"

// Expose private members intentionally so we can verify setter side-effects
// and directly unit-test RCPreprocessImg / LDPreprocessImg.
#define private public
#include <inference/orchestrator.hpp>
#undef private

#include <NvInfer.h>
#include <opencv2/opencv.hpp>

using namespace Inference;

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
// Fixture shared by all Orchestrator tests
// ============================================================

class OrchestratorTest : public ::testing::Test {
protected:
    Orchestrator orc;
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
    EXPECT_EQ(Orchestrator::GetMemorySize(dims, p.element_size), p.expected);
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
    EXPECT_EQ(Orchestrator::GetMemorySize(dims, sizeof(float)), 0UL);
}

// ============================================================
// Setters — bool flags
// ============================================================

TEST_F(OrchestratorTest, SetPreloadRCEngine_UpdatesFlag)
{
    orc.SetPreloadRCEngine(true);  EXPECT_TRUE(orc.preload_rc_engine_);
    orc.SetPreloadRCEngine(false); EXPECT_FALSE(orc.preload_rc_engine_);
}

TEST_F(OrchestratorTest, SetPreloadLDEngines_UpdatesFlag)
{
    orc.SetPreloadLDEngines(true);  EXPECT_TRUE(orc.preload_ld_engines_);
    orc.SetPreloadLDEngines(false); EXPECT_FALSE(orc.preload_ld_engines_);
}

TEST_F(OrchestratorTest, SetUseTRTForLD_UpdatesConfigFlag)
{
    orc.SetUseTRTForLD(true);  EXPECT_TRUE(orc.ldnet_config.use_trt);
    orc.SetUseTRTForLD(false); EXPECT_FALSE(orc.ldnet_config.use_trt);
}

// ============================================================
// Setters — path validation (bad path → silent reject)
// ============================================================

TEST_F(OrchestratorTest, SetRCNetEnginePath_InvalidPath_Rejected)
{
    std::string original = orc.rc_engine_path_;
    EXPECT_EQ(orc.SetRCNetEnginePath("/bad/path/engine.trt"), EC::FILE_DOES_NOT_EXIST);
    EXPECT_EQ(orc.rc_engine_path_, original);
}

TEST_F(OrchestratorTest, SetRCNetEnginePath_EmptyPath_Rejected)
{
    std::string original = orc.rc_engine_path_;
    EXPECT_EQ(orc.SetRCNetEnginePath(""), EC::FILE_DOES_NOT_EXIST);
    EXPECT_EQ(orc.rc_engine_path_, original);
}

TEST_F(OrchestratorTest, SetLDNetEngineFolderPath_InvalidPath_Rejected)
{
    std::string original = orc.ld_engine_folder_path_;
    EXPECT_EQ(orc.SetLDNetEngineFolderPath("/bad/folder"), EC::FILE_DOES_NOT_EXIST);
    EXPECT_EQ(orc.ld_engine_folder_path_, original);
}

// ============================================================
// SetLDNetConfig
// ============================================================

TEST_F(OrchestratorTest, SetLDNetConfig_AllFieldsUpdated)
{
    orc.SetLDNetConfig(NET_QUANTIZATION::INT8, 1024, 768, true, false);
    EXPECT_EQ(orc.ldnet_config.weight_quant, NET_QUANTIZATION::INT8);
    EXPECT_EQ(orc.ldnet_config.input_width,  1024);
    EXPECT_EQ(orc.ldnet_config.input_height, 768);
    EXPECT_TRUE(orc.ldnet_config.embedded_nms);
    EXPECT_FALSE(orc.ldnet_config.use_trt);
    // Cross-check via appendix so the struct is actually being used correctly
    EXPECT_EQ(orc.ldnet_config.GetFileNameAppendix(), "_weights_int8_sz_1024_nms.onnx");
}

TEST_F(OrchestratorTest, SetLDNetConfig_CanBeOverwritten)
{
    orc.SetLDNetConfig(NET_QUANTIZATION::FP16, 4608, 2592, false, true);
    orc.SetLDNetConfig(NET_QUANTIZATION::FP32, 2048, 2048, true,  false);
    EXPECT_EQ(orc.ldnet_config.weight_quant, NET_QUANTIZATION::FP32);
    EXPECT_EQ(orc.ldnet_config.input_width,  2048);
}

// ============================================================
// GrabNewImage
// ============================================================

TEST_F(OrchestratorTest, GrabNewImage_NullFrame_IsNoOp)
{
    EXPECT_NO_FATAL_FAILURE(orc.GrabNewImage(nullptr));
    EXPECT_EQ(orc.original_frame_, nullptr);
}

TEST_F(OrchestratorTest, GrabNewImage_ValidFrame_SetsPointerAndResetsCounters)
{
    orc.num_rc_inferences_on_current_frame_ = 5;
    orc.num_ld_inferences_on_current_frame_ = 3;
    auto frame = MakeSyntheticFrame();
    orc.GrabNewImage(frame);

    EXPECT_EQ(orc.original_frame_, frame);
    EXPECT_EQ(orc.num_rc_inferences_on_current_frame_, 0);
    EXPECT_EQ(orc.num_ld_inferences_on_current_frame_, 0);
}

TEST_F(OrchestratorTest, GrabNewImage_SuccessiveCallsReplaceFrame)
{
    orc.GrabNewImage(MakeSyntheticFrame(320, 240));
    auto frame2 = MakeSyntheticFrame(640, 480);
    orc.GrabNewImage(frame2);
    EXPECT_EQ(orc.original_frame_, frame2);
}

// ============================================================
// ExecRCInference — error paths
// ============================================================

TEST_F(OrchestratorTest, ExecRCInference_NoFrame_ReturnsNoFrameError)
{
    EXPECT_EQ(orc.ExecRCInference(), EC::NN_NO_FRAME_AVAILABLE);
}

// Both preload modes should end in NN_ENGINE_NOT_INITIALIZED when no engine is loaded.
// The constructor may load the real engine if default paths exist, so we explicitly
// unload it and redirect rc_engine_path_ to a non-existent file via the private member.
TEST_F(OrchestratorTest, ExecRCInference_EngineAbsent_PreloadOn_ReturnsEngineNotInitialized)
{
    orc.FreeRCNet();
    orc.rc_engine_path_ = "/nonexistent/path.trt"; // bypass filesystem-validated setter
    orc.SetPreloadRCEngine(true);
    orc.GrabNewImage(MakeSyntheticFrame());
    // preload=true → skips on-demand load, rc_net_ uninitialized → engine-not-init error
    EXPECT_EQ(orc.ExecRCInference(), EC::NN_ENGINE_NOT_INITIALIZED);
}

TEST_F(OrchestratorTest, ExecRCInference_EngineAbsent_PreloadOff_ReturnsEngineNotInitialized)
{
    orc.FreeRCNet();
    orc.rc_engine_path_ = "/nonexistent/path.trt"; // bypass filesystem-validated setter
    orc.SetPreloadRCEngine(false);
    orc.GrabNewImage(MakeSyntheticFrame());
    // preload=false → tries on-demand load from bad path, fails → engine-not-init error
    EXPECT_EQ(orc.ExecRCInference(), EC::NN_ENGINE_NOT_INITIALIZED);
}

// ============================================================
// ExecLDInference — error paths
// ============================================================

TEST_F(OrchestratorTest, ExecLDInference_NoFrame_ReturnsNoFrameError)
{
    EXPECT_EQ(orc.ExecLDInference(), EC::NN_NO_FRAME_AVAILABLE);
}

TEST_F(OrchestratorTest, ExecLDInference_FrameWithNoRegions_ReturnsOK)
{
    // Frame has no RC detections → LD short-circuits cleanly
    orc.GrabNewImage(MakeSyntheticFrame());
    EXPECT_EQ(orc.ExecLDInference(), EC::OK);
}

TEST_F(OrchestratorTest, ExecFullInference_NoFrame_ReturnsNoFrameError)
{
    EXPECT_EQ(orc.ExecFullInference(), EC::NN_NO_FRAME_AVAILABLE);
}

// ============================================================
// Free methods — must not crash regardless of state
// ============================================================

TEST_F(OrchestratorTest, FreeAll_DoesNotCrash)
{
    EXPECT_NO_FATAL_FAILURE(orc.FreeEngines());
    EXPECT_NO_FATAL_FAILURE(orc.FreeRCNet());
    EXPECT_NO_FATAL_FAILURE(orc.FreeLDNets());
    EXPECT_NO_FATAL_FAILURE(orc.FreeEngines()); // double-free safe
}

TEST_F(OrchestratorTest, FreeLDNetForRegion_RegionNotInMap_DoesNotCrash)
{
    // ld_nets_ is empty (no model folder found) — must be a silent no-op
    EXPECT_NO_FATAL_FAILURE(orc.FreeLDNetForRegion(RegionID::R_17T));
}

// ============================================================
// Private: RCPreprocessImg
// ============================================================

TEST_F(OrchestratorTest, RCPreprocessImg_EmptyImage_OutputReleased)
{
    cv::Mat output(10, 10, CV_32F, cv::Scalar(1.0f));
    orc.RCPreprocessImg(cv::Mat{}, output);
    EXPECT_TRUE(output.empty());
}

TEST_F(OrchestratorTest, RCPreprocessImg_ValidBGR_ProducesCorrectBlob)
{
    cv::Mat output;
    orc.RCPreprocessImg(cv::Mat(480, 640, CV_8UC3, cv::Scalar(100, 150, 200)), output);
    EXPECT_FALSE(output.empty());
    EXPECT_EQ(output.depth(), CV_32F);
    EXPECT_EQ(output.total(), 1UL * 3 * 224 * 224); // NCHW blob
}

TEST_F(OrchestratorTest, RCPreprocessImg_ValuesAreImageNetNormalised)
{
    cv::Mat output;
    orc.RCPreprocessImg(cv::Mat(224, 224, CV_8UC3, cv::Scalar(128, 128, 128)), output);
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
    orc.RCPreprocessImg(cv::Mat(h, w, CV_8UC3, cv::Scalar(50, 100, 200)), output);
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
    orc.LDPreprocessImg(cv::Mat(256, 256, CV_8UC3, cv::Scalar(255, 255, 255)), output, 256, 256);
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
    orc.LDPreprocessImg(cv::Mat(p.src_h, p.src_w, CV_8UC3, cv::Scalar(50, 100, 150)), output, p.tgt_w, p.tgt_h);
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
    if (orc.ld_nets_.empty())
    {
        GTEST_SKIP() << "No LDNet runtimes initialised (missing model assets). Skipping.";
    }
    RegionID test_region = orc.ld_nets_.begin()->first;
    ASSERT_EQ(orc.LoadLDNetEngineForRegion(test_region), EC::OK);

    size_t free_before, free_after;
    cudaMemGetInfo(&free_before, &total);

    // Changing config must free the loaded engine before rebuilding.
    orc.SetLDNetConfig(orc.ldnet_config.weight_quant,
                       orc.ldnet_config.input_width,
                       orc.ldnet_config.input_height,
                       orc.ldnet_config.embedded_nms,
                       orc.ldnet_config.use_trt);

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
    if (orc.ld_nets_.empty())
        GTEST_SKIP() << "No LDNet runtimes available (missing model assets).";

    RegionID region = orc.ld_nets_.begin()->first;
    // Force the guard to fire: require SIZE_MAX/2 bytes beyond the 2.5x estimate.
    // The check is overflow-safe, so this always exceeds real free memory.
    orc.ld_nets_[region]->gpu_reserve_bytes_ = SIZE_MAX / 2;

    EXPECT_EQ(orc.LoadLDNetEngineForRegion(region), EC::NN_INSUFFICIENT_GPU_MEMORY);
    EXPECT_FALSE(orc.ld_nets_[region]->IsInitialized()); // no engine was loaded

    orc.ld_nets_[region]->gpu_reserve_bytes_ = 0; // restore
}

TEST_F(OrchestratorTest, EnsureScratchBuffers_InsufficientGPUMemory_ReturnsError)
{
    // Needs one real engine load to get past the initialized/context check.
    constexpr size_t kMinFree = 128ULL * 1024 * 1024;
    size_t gpu_free, gpu_total;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    if (gpu_free < kMinFree)
        GTEST_SKIP() << "Only " << (gpu_free >> 20) << " MiB GPU free — need 128 MiB to load one engine.";
    if (orc.ld_nets_.empty())
        GTEST_SKIP() << "No LDNet runtimes available (missing model assets).";

    RegionID region = orc.ld_nets_.begin()->first;
    ASSERT_EQ(orc.LoadLDNetEngineForRegion(region), EC::OK);

    orc.ld_nets_[region]->gpu_reserve_bytes_ = SIZE_MAX / 2;
    EXPECT_EQ(orc.ld_nets_[region]->EnsureScratchBuffers(), EC::NN_INSUFFICIENT_GPU_MEMORY);

    orc.ld_nets_[region]->gpu_reserve_bytes_ = 0;
    orc.FreeLDNetForRegion(region);
}

TEST_F(OrchestratorTest, LoadLDNetEngines_LowMemory_StopsBeforeFirstLoad)
{
    // Drive the between-load threshold above any realistic free memory so
    // LoadLDNetEngines() halts immediately without touching the GPU.
    orc.min_gpu_free_between_loads_ = SIZE_MAX / 2;
    orc.SetPreloadLDEngines(true);
    orc.LoadLDNetEngines();

    for (const auto& [region_id, ld_net] : orc.ld_nets_)
    {
        EXPECT_FALSE(ld_net->IsInitialized())
            << "Region " << GetRegionString(region_id) << " should not be loaded under simulated low GPU memory.";
    }

    orc.min_gpu_free_between_loads_ = 256ULL * 1024 * 1024; // restore
}
