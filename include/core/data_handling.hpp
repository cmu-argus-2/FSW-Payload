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



#define ROOT_DATA_FOLDER "data/"
#define IMAGES_FOLDER "data/images/"
#define TELEMETRY_FOLDER "data/telemetry/"
#define EXPERIMENTS_FOLDER "data/experiments/"
#define LOGGING_FOLDER "data/logging/"

#define DELIMITER "_"

namespace DH // Data Handling
{

    /*
        Initialize the folder structure for all data storage activities.
        The structure is as follows:
        data/
        ├── images/
        ├── telemetry/
        ├── experiments/ (orbit determination results)
        ├── logging/

        @return true if successful, false otherwise
    */
    bool InitializeDataStorage();



    bool make_directory(std::string_view directory_path);
    long GetFileSize(std::string_view file_path);
    long GetDirectorySize(std::string_view directory_path); 
    int CountFilesInDirectory(std::string_view directory_path);

    // raw_timestamp_camid.png - timestamp should be more accurate 
    void StoreRawImgToDisk(std::uint64_t timestamp, int cam_id, const cv::Mat& img);

    // Load in memory the latest img
    bool ReadLatestStoredRawImg(cv::Mat& img, std::uint64_t& timestamp, int& cam_id);

    // count the number of images in the images folder
    int CountRawImgNumberOnDisk();



    // need to read from disk (transfre img from disk to RAM) - reinitialize the frame buffer
    // need to be able to select the latest img from the disk
    // need to save telemetry data on disk


}







#endif // DATA_HANDLING_HPP