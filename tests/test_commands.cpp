#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <cstdint>

// Expose private members so we can access DatasetManager::current_dataset
// for folder path cleanup. Must appear before class headers.
#define private public

#include "commands.hpp"
#include "payload.hpp"
#include "vision/dataset_manager.hpp"
#include "vision/dataset.hpp"
#include "communication/named_pipe.hpp"
#include "messages.hpp"
#include "configuration.hpp"
#include "core/timing.hpp"

namespace fs = std::filesystem;

/*
TODO
- Check total size of commands array
- Check uniqueness of command IDs
- check if all commands are declared 
- check contguity of command IDs (with total size)



*/

// DATASET_KEY_CMD is #define'd in commands.cpp (not the header).
static constexpr const char* DS_CMD_KEY = "CMD";

// ── Helpers ───────────────────────────────────────────────────────────────────

static void DrainTxQueue()
{
    while (!sys::payload().GetTxQueue().IsEmpty())
        sys::payload().GetTxQueue().GetNextMsg();
}

static void ClearActiveDatasets()
{
    for (const auto& key : DatasetManager::ListActiveDatasetManagers())
        DatasetManager::StopDatasetManager(key);
}

// ACK packet layout: [cmd_id][seq_hi][seq_lo][len_hi][len_lo][status_byte]
static uint8_t AckStatus(const std::shared_ptr<Message>& msg)
{
    EXPECT_GE(msg->packet.size(), 6u) << "Packet too short to have a status byte";
    return msg->packet[5];
}

static std::shared_ptr<Message> PopMsg()
{
    EXPECT_FALSE(sys::payload().GetTxQueue().IsEmpty()) << "TX queue unexpectedly empty";
    return sys::payload().GetTxQueue().GetNextMsg();
}

// Builds a valid 13-byte START_CAPTURE_DATASET payload.
// start_time_s=0 forces the command to clamp to current time.
static std::vector<uint8_t> ValidStartData(uint32_t start_time_s = 0)
{
    std::vector<uint8_t> d(13, 0);
    d[0]  = static_cast<uint8_t>(CAPTURE_MODE::PERIODIC);          // capture_mode
    d[1]  = 0; d[2] = 60;                                           // max_period = 60 s
    d[3]  = 0; d[4] = 10;                                           // target_frame_nb = 10
    d[5]  = (start_time_s >> 24) & 0xFF;
    d[6]  = (start_time_s >> 16) & 0xFF;
    d[7]  = (start_time_s >>  8) & 0xFF;
    d[8]  =  start_time_s        & 0xFF;
    d[9]  = static_cast<uint8_t>(IMU_COLLECTION_MODE::NONE);        // imu_collection_mode
    d[10] = 1;                                                       // image_capture_rate = 1 s
    d[11] = 10;                                                      // imu_sample_rate = 1.0 Hz
    d[12] = static_cast<uint8_t>(ProcessingStage::NotPrefiltered);  // target_processing_stage
    return d;
}

// ── Fixture ───────────────────────────────────────────────────────────────────

struct CommandTest : ::testing::Test
{
    // Payload is a Meyers singleton — constructed exactly once for the process.
    // Configuration() leaves camera/IMU fields zero-initialised (no file I/O).
    // NamedPipe::Connect() is never called because we don't call Payload::Run().
    static void SetUpTestSuite()
    {
        Payload::CreateInstance(
            std::make_unique<Configuration>(),
            std::make_unique<NamedPipe>()
        );
    }

    void SetUp() override
    {
        DrainTxQueue();
        ClearActiveDatasets();
    }

    void TearDown() override
    {
        ClearActiveDatasets();
        DrainTxQueue();
        for (const auto& f : test_folders)
            fs::remove_all(f);
    }

    std::vector<std::string> test_folders;
};

// ── start_capture_dataset: data length guard ──────────────────────────────────

TEST_F(CommandTest, StartDataset_EmptyData_Rejected)
{
    std::vector<uint8_t> data;
    start_capture_dataset(data);
    auto msg = PopMsg();
    EXPECT_EQ(msg->packet[0], CommandID::START_CAPTURE_DATASET);
    EXPECT_EQ(AckStatus(msg), 0x20);
}

TEST_F(CommandTest, StartDataset_ShortData_Rejected)
{
    // 12 bytes — one short of the required 13
    std::vector<uint8_t> data(12, 0x02);
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

// ── start_capture_dataset: capture_mode byte (data[0]) ───────────────────────

TEST_F(CommandTest, StartDataset_CaptureModeIDLE_Rejected)
{
    // IDLE (0) is below the valid floor of PERIODIC (2)
    auto data = ValidStartData();
    data[0] = static_cast<uint8_t>(CAPTURE_MODE::IDLE);
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

TEST_F(CommandTest, StartDataset_CaptureModeSingle_Rejected)
{
    // CAPTURE_SINGLE (1) is also below PERIODIC (2)
    auto data = ValidStartData();
    data[0] = static_cast<uint8_t>(CAPTURE_MODE::CAPTURE_SINGLE);
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

TEST_F(CommandTest, StartDataset_CaptureModeOutOfRange_Rejected)
{
    auto data = ValidStartData();
    data[0] = 99;
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

// ── start_capture_dataset: imu_collection_mode byte (data[9]) ────────────────

TEST_F(CommandTest, StartDataset_InvalidIMUMode_Rejected)
{
    auto data = ValidStartData();
    data[9] = 99; // above GYRO_MAG_TEMP
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

// ── start_capture_dataset: target_processing_stage byte (data[12]) ───────────

TEST_F(CommandTest, StartDataset_InvalidProcessingStage_Rejected)
{
    auto data = ValidStartData();
    data[12] = 99; // above LDNeted
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

// ── start_capture_dataset: target_frame_nb (data[3..4]) ──────────────────────

TEST_F(CommandTest, StartDataset_FrameNbZero_Rejected)
{
    auto data = ValidStartData();
    data[3] = 0; data[4] = 0;
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

TEST_F(CommandTest, StartDataset_FrameNbOverflow_Rejected)
{
    // 256 (0x0100) > MAX_SAMPLES (255) — the old code silently truncated this to 0
    auto data = ValidStartData();
    data[3] = 0x01; data[4] = 0x00;
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

TEST_F(CommandTest, StartDataset_FrameNbMaxValid_Accepted)
{
    // 255 == MAX_SAMPLES — boundary that must pass the range check.
    // Future start time keeps the thread in loop_cv.wait_until, avoiding
    // Argus initialisation in the test environment (see PassesValidationFailsAtRuntime).
    const uint32_t future_s = static_cast<uint32_t>(timing::GetCurrentTimeMs() / 1000 + 60);
    auto data = ValidStartData(future_s);
    data[3] = 0x00; data[4] = 0xFF;
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), ACK_SUCCESS);
    auto dm = DatasetManager::GetActiveDatasetManager(DS_CMD_KEY);
    if (dm) test_folders.push_back(dm->current_dataset.GetFolderPath());
}

// ── start_capture_dataset: semantic checks via isValidConfiguration ───────────

TEST_F(CommandTest, StartDataset_PeriodZero_Rejected)
{
    // max_period = 0 — below ABSOLUTE_MINIMUM_PERIOD (0.1 s)
    auto data = ValidStartData();
    data[1] = 0; data[2] = 0;
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

TEST_F(CommandTest, StartDataset_IMURateZero_Rejected)
{
    // data[11] = 0 → imu_sample_rate_hz = 0.0 Hz — fails isValidConfiguration
    auto data = ValidStartData();
    data[11] = 0;
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

TEST_F(CommandTest, StartDataset_IMURateExceedsMax_Rejected)
{
    // data[11] = 251 → 25.1 Hz > 25.0 Hz limit
    auto data = ValidStartData();
    data[11] = 251;
    start_capture_dataset(data);
    EXPECT_EQ(AckStatus(PopMsg()), 0x20);
}

// ── start_capture_dataset: valid payload distinguishes validation from runtime ─

TEST_F(CommandTest, StartDataset_ValidData_PassesValidationFailsAtRuntime)
{
    // All 13 bytes are valid — the command must pass every validation check and
    // return ACK_SUCCESS. We use a future start time so the collection thread
    // sits in loop_cv.wait_until rather than calling PrepareForCapture, which
    // would initialise the Argus camera library and hang the test process.
    // TearDown cleans up via StopDatasetManager, which interrupts the wait.
    const uint32_t future_s = static_cast<uint32_t>(timing::GetCurrentTimeMs() / 1000 + 60);
    auto data = ValidStartData(future_s);
    start_capture_dataset(data);
    auto msg = PopMsg();
    EXPECT_EQ(msg->packet[0], CommandID::START_CAPTURE_DATASET);
    EXPECT_EQ(AckStatus(msg), ACK_SUCCESS);

    auto dm = DatasetManager::GetActiveDatasetManager(DS_CMD_KEY);
    if (dm) test_folders.push_back(dm->current_dataset.GetFolderPath());
}

// ── stop_capture_dataset ──────────────────────────────────────────────────────

TEST_F(CommandTest, StopDataset_NoneRunning_ErrorAck)
{
    std::vector<uint8_t> data;
    stop_capture_dataset(data);
    auto msg = PopMsg();
    EXPECT_EQ(msg->packet[0], CommandID::STOP_CAPTURE_DATASET);
    EXPECT_EQ(AckStatus(msg), 0x22);
}

TEST_F(CommandTest, StopDataset_RegisteredDataset_SuccessAck)
{
    // Create a dataset in the registry without calling StartCollection,
    // then stop it via the command. StopDatasetManager handles the rest.
    uint64_t start_ms = timing::GetCurrentTimeMs() + 60000;
    auto dm = DatasetManager::Create(
        60.0, 5, CAPTURE_MODE::PERIODIC, start_ms,
        IMU_COLLECTION_MODE::NONE, 1, 1.0f,
        ProcessingStage::NotPrefiltered, DS_CMD_KEY,
        sys::cameraManager(), sys::imuManager(), sys::inferenceManager()
    );
    test_folders.push_back(dm->current_dataset.GetFolderPath());

    std::vector<uint8_t> data;
    stop_capture_dataset(data);

    auto msg = PopMsg();
    EXPECT_EQ(msg->packet[0], CommandID::STOP_CAPTURE_DATASET);
    EXPECT_EQ(AckStatus(msg), ACK_SUCCESS);
    EXPECT_EQ(DatasetManager::GetActiveDatasetManager(DS_CMD_KEY), nullptr);
}

TEST_F(CommandTest, StopDataset_CalledTwice_SecondCallIsError)
{
    // First stop succeeds; second stop must not crash and must return error.
    uint64_t start_ms = timing::GetCurrentTimeMs() + 60000;
    auto dm = DatasetManager::Create(
        60.0, 5, CAPTURE_MODE::PERIODIC, start_ms,
        IMU_COLLECTION_MODE::NONE, 1, 1.0f,
        ProcessingStage::NotPrefiltered, DS_CMD_KEY,
        sys::cameraManager(), sys::imuManager(), sys::inferenceManager()
    );
    test_folders.push_back(dm->current_dataset.GetFolderPath());

    std::vector<uint8_t> data;
    stop_capture_dataset(data);
    stop_capture_dataset(data);

    EXPECT_EQ(AckStatus(PopMsg()), ACK_SUCCESS);
    EXPECT_EQ(AckStatus(PopMsg()), 0x22);
}
