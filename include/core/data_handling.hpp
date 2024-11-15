/*

This file contains classes and functions providing file services and data handling to the flight software.

*/

#ifndef DATA_HANDLING_HPP
#define DATA_HANDLING_HPP

#include <filesystem>
#include <string_view>
#include "spdlog/spdlog.h"


#define ROOT_DATA_FOLDER "data/"
#define IMAGES_FOLDER "data/images/"
#define TELEMETRY_FOLDER "data/telemetry/"
#define EXPERIMENTS_FOLDER "data/experiments/"
#define LOGGING_FOLDER "data/logging/"


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


}







#endif // DATA_HANDLING_HPP