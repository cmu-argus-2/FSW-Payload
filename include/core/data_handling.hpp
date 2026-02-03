/*

This file contains classes and functions providing file services and data handling to the flight software.

*/

#ifndef DATA_HANDLING_HPP
#define DATA_HANDLING_HPP

#include <filesystem>
#include <cstdint>
#include <string_view>
#include <vector>
#include <opencv2/opencv.hpp>
#include "spdlog/spdlog.h"
#include "vision/frame.hpp"

#include "core/errors.hpp"

#define ROOT_DISK "/"

#define ROOT_DATA_FOLDER "data/"
#define IMAGES_FOLDER "data/images/"
#define TELEMETRY_FOLDER "data/telemetry/"
#define DATASETS_FOLDER "data/datasets/"
#define RESULTS_FOLDER "data/results/"
#define LOGGING_FOLDER "data/logging/"
#define COMMS_FOLDER "data/comms/"


#define DELIMITER "_"

namespace DH // Data Handling
{
    namespace fs = std::filesystem;
    constexpr std::uint16_t DH_PACKET_HEADER_SIZE = 2;
    constexpr std::uint16_t DH_MAX_PAYLOAD_SIZE = 240;
    constexpr std::uint16_t DH_FIXED_PACKET_SIZE = DH_PACKET_HEADER_SIZE + DH_MAX_PAYLOAD_SIZE; // 242 bytes on disk
    constexpr std::uint16_t DH_FILE_HEADER_SIZE = 5; // magic number length

    /*
        Initialize the folder structure for all data storage activities.
        The structure is as follows:
        data/
        ├── images/
        ├── telemetry/
        ├── datasets/
        ├── results/ (orbit determination results)
        ├── logging/
        └── comms/ (Special folder for communication purposes)

        @return true if successful, false otherwise
    */
    bool InitializeDataStorage();


    bool MakeNewDirectory(std::string_view directory_path);
    long GetFileSize(std::string_view file_path); // in bytes
    long GetDirectorySize(std::string_view directory_path); 
    int CountFilesInDirectory(std::string_view directory_path);


    // raw_timestamp_camid.png 
    std::string StoreFrameToDisk(Frame& frame, std::string_view target_folder = IMAGES_FOLDER);
    std::string StoreRawImgToDisk(std::uint64_t timestamp, int cam_id, const cv::Mat& img, std::string_view target_folder = IMAGES_FOLDER);
    void StoreFrameMetadataToDisk(Frame& frame, std::string_view target_folder = IMAGES_FOLDER);

    // Load in memory the latest img
    bool ReadLatestStoredRawImg(Frame& frame);
    bool ReadHighestValueStoredRawImg(Frame& frame);
    
    // Load an image from disk by its path
    bool ReadImageFromDisk(const std::string& file_path, Frame& frame_out);

    // Load the latest binary image file with packet header (img_<timestamp>_<camera_id>.bin)
    // Returns the file path if found, empty string otherwise
    std::string GetLatestImgBinPath();
    
    // Load in memory a chunk of data
    EC ReadFileChunk(std::string_view file_path, uint32_t start_byte, uint32_t length, std::vector<uint8_t>& data_out);

    // count the number of images in the images folder
    int CountRawImgNumberOnDisk();



    // need to read from disk (transfre img from disk to RAM) - reinitialize the frame buffer
    // need to be able to select the latest img from the disk
    // need to save telemetry data on disk



    // Returns true if the latest file is found, false otherwise
    bool GetLatestRawFilePath(std::filesystem::directory_entry& latest_file);
    bool GetHighestValueRawFilePath(std::filesystem::directory_entry& highest_value_file);

    /*
    Assumes the primary system disk is an NVMe drive as the main root partition (/)
    Returns the disk usage as a positive percentage. -1 in case of errors.
    */
    int GetTotalDiskUsage();


    // Communication related helper functions
    void EmptyCommsFolder();
    std::string CopyFrameToCommsFolder(Frame& frame);
    EC GetCommsFilePath(std::string& path_out);

    // Data handler formatted fixed-packet files (matches Python FileProcess)
    bool WriteFixedPacketFile(const std::string& output_path, const std::vector<std::vector<uint8_t>>& payloads);
    bool ReadFixedPacketFile(const std::string& input_path, std::vector<std::vector<uint8_t>>& payloads_out);



}







#endif // DATA_HANDLING_HPP
