/*

This file contains classes and functions providing file services and data handling to the flight software.

*/

#ifndef DATA_HANDLING_HPP
#define DATA_HANDLING_HPP

#include <filesystem>
#include <string_view>
#include <vector>
#include <opencv2/opencv.hpp>
#include "spdlog/spdlog.h"
#include "vision/frame.hpp"

#define ROOT_DISK "/"

#define ROOT_DATA_FOLDER "data/"
#define IMAGES_FOLDER "data/images/"
#define TELEMETRY_FOLDER "data/telemetry/"
#define DATASETS_FOLDER "data/datasets/"
#define RESULTS_FOLDER "data/results/"
#define LOGGING_FOLDER "data/logging/"


#define DELIMITER "_"

namespace DH // Data Handling
{
    namespace fs = std::filesystem;

    /*
        Initialize the folder structure for all data storage activities.
        The structure is as follows:
        data/
        ├── images/
        ├── telemetry/
        ├── datasets/
        ├── results/ (orbit determination results)
        ├── logging/

        @return true if successful, false otherwise
    */
    bool InitializeDataStorage();



    bool MakeNewDirectory(std::string_view directory_path);
    long GetFileSize(std::string_view file_path);
    long GetDirectorySize(std::string_view directory_path); 
    int CountFilesInDirectory(std::string_view directory_path);


    

    // raw_timestamp_camid.png 
    void StoreFrameToDisk(Frame& frame, std::string_view target_folder = IMAGES_FOLDER);
    void StoreRawImgToDisk(std::uint64_t timestamp, int cam_id, const cv::Mat& img, std::string_view target_folder = IMAGES_FOLDER);

    // Load in memory the latest img
    bool ReadLatestStoredRawImg(Frame& frame);

    // count the number of images in the images folder
    int CountRawImgNumberOnDisk();



    // need to read from disk (transfre img from disk to RAM) - reinitialize the frame buffer
    // need to be able to select the latest img from the disk
    // need to save telemetry data on disk



    /*
    Assumes the primary system disk is an NVMe drive as the main root partition (/)
    Returns the disk usage as a positive percentage. -1 in case of errors.
    */
    int GetTotalDiskUsage();


}







#endif // DATA_HANDLING_HPP