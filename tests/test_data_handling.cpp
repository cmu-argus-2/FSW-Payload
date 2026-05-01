#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <system_error>
#include "core/data_handling.hpp"

namespace fs = std::filesystem;

namespace
{
std::string UniqueTestDirName()
{
    return "payload_dh_test_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed())
         + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
}

Frame MakeFrame(int cam_id, std::uint64_t timestamp, const cv::Scalar& color)
{
    cv::Mat img(16, 16, CV_8UC3, color);
    return Frame(cam_id, img, timestamp);
}
}

class DataHandlingPathTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        old_cwd = fs::current_path();
        temp_root = fs::temp_directory_path() / UniqueTestDirName();
        fs::create_directories(temp_root / IMAGES_FOLDER);
        fs::create_directories(temp_root / COMMS_FOLDER);
        fs::current_path(temp_root);
    }

    void TearDown() override
    {
        fs::current_path(old_cwd);
        std::error_code ec;
        fs::remove_all(temp_root, ec);
    }

    fs::path old_cwd;
    fs::path temp_root;
};

TEST_F(DataHandlingPathTest, ReadLatestStoredRawImgPrefersRawImagesOverMetadata)
{
    Frame frame_in = MakeFrame(1, 1234, cv::Scalar(10, 20, 30));
    DH::StoreFrameToDisk(frame_in);

    Frame frame_out;
    ASSERT_TRUE(DH::ReadLatestStoredRawImg(frame_out));
    EXPECT_EQ(frame_out.GetTimestamp(), frame_in.GetTimestamp());
    EXPECT_EQ(frame_out.GetCamID(), frame_in.GetCamID());
}

TEST_F(DataHandlingPathTest, ReadHighestValueStoredRawImgUsesMetadataRawImagePath)
{
    Frame earlier = MakeFrame(0, 1000, cv::Scalar(0, 0, 0));
    Frame later   = MakeFrame(0, 2000, cv::Scalar(255, 255, 255));
    DH::StoreFrameToDisk(earlier);
    DH::StoreFrameToDisk(later);

    Frame frame_out;
    ASSERT_TRUE(DH::ReadHighestValueStoredRawImg(frame_out));
    EXPECT_EQ(frame_out.GetTimestamp(), later.GetTimestamp());
    EXPECT_EQ(frame_out.GetCamID(), later.GetCamID());
}

TEST_F(DataHandlingPathTest, StoreFrameToDisk_DefaultsToJpgAndPersistsRawImageMetadata)
{
    Frame frame = MakeFrame(3, 3333, cv::Scalar(1, 2, 3));

    const std::string img_path = DH::StoreFrameToDisk(frame);
    ASSERT_FALSE(img_path.empty());
    EXPECT_EQ(fs::path(img_path).extension(), ".jpg");
    EXPECT_TRUE(fs::exists(img_path));

    Json metadata = DH::LoadFrameMetadataFromDisk(frame.GetTimestamp(), frame.GetCamID());
    ASSERT_TRUE(metadata.is_object());
    ASSERT_TRUE(metadata.contains("raw_image"));
    ASSERT_TRUE(metadata["raw_image"].is_object());
    EXPECT_EQ(metadata["raw_image"]["format"], "JPG");
    EXPECT_EQ(fs::path(metadata["raw_image"]["path"].get<std::string>()).extension(), ".jpg");
}

TEST_F(DataHandlingPathTest, ReadImageFromDiskByTimestampFindsDefaultJpg)
{
    Frame frame_in = MakeFrame(2, 2222, cv::Scalar(50, 60, 70));
    ASSERT_FALSE(DH::StoreFrameToDisk(frame_in).empty());

    Frame frame_out;
    ASSERT_TRUE(DH::ReadImageFromDisk(frame_in.GetTimestamp(), frame_in.GetCamID(), frame_out));
    EXPECT_EQ(frame_out.GetTimestamp(), frame_in.GetTimestamp());
    EXPECT_EQ(frame_out.GetCamID(), frame_in.GetCamID());
    EXPECT_EQ(frame_out.GetImg().size(), frame_in.GetImg().size());
}

TEST_F(DataHandlingPathTest, StoreFrameToDiskMissingFolderReturnsEmptyAndSkipsMetadata)
{
    Frame frame = MakeFrame(0, 4444, cv::Scalar(9, 8, 7));
    const fs::path missing_dir = temp_root / "missing";

    const std::string img_path = DH::StoreFrameToDisk(frame, missing_dir.string());
    EXPECT_TRUE(img_path.empty());
    EXPECT_FALSE(fs::exists(missing_dir / "frame_4444_0.json"));
}

TEST_F(DataHandlingPathTest, GetCommsFilePathReturnsLatestFileFromCommsFolder)
{
    Frame frame = MakeFrame(2, 4321, cv::Scalar(5, 15, 25));
    const std::string stored_path = DH::CopyFrameToCommsFolder(frame);

    std::string comms_path;
    ASSERT_EQ(DH::GetCommsFilePath(comms_path), EC::OK);
    EXPECT_EQ(fs::weakly_canonical(comms_path), fs::weakly_canonical(stored_path));
}
