/*

This file contains classes and functions providing file services and data handling to the flight software.

*/

#ifndef DATA_HANDLING_HPP
#define DATA_HANDLING_HPP

#include <filesystem>
#include <string_view>
#include "spdlog/spdlog.h"


bool make_directory(std::string_view directory_path);
long GetFileSize(std::string_view file_path);
long GetDirectorySize(std::string_view directory_path);


class DataHandler
{

private:
    bool init_data_folder_tree = false;
    const std::string f_root = "data/";
    const std::string f_images = f_root + "images/";
    const std::string f_telemetry = f_root + "telemetry/";
    const std::string f_experiments = f_root + "experiments/";
    const std::string f_logging = f_root + "logging/";


public:

    DataHandler();
    // ~DataHandler() = default;

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

};





#endif // DATA_HANDLING_HPP