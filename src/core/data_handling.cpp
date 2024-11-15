#include "core/data_handling.hpp"


namespace DH // Data Handling
{

bool INIT_DATA_FOLDER_TREE = false;

bool make_new_directory(std::string_view directory_path)
{
    bool success = false;
    if (std::filesystem::exists(directory_path)) {
        SPDLOG_INFO("Folder already exists.");
        success = true;
    } else if (std::filesystem::create_directory(directory_path)) {
        SPDLOG_INFO("Data folder created.");
        success = true;
    } else {
        SPDLOG_CRITICAL("Failed to create data folder.");
    }    

    return success;

}


long GetFileSize(std::string_view file_path) 
{
    struct stat stat_buf;
    int rc = stat(std::string(file_path).c_str(), &stat_buf);
    return rc == 0LL ? stat_buf.st_size : -1LL;
}


long GetDirectorySize(std::string_view directory_path) 
{
    long total_size = 0;
    struct stat stat_buf;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            if (stat(entry.path().c_str(), &stat_buf) == 0) {
                total_size += stat_buf.st_size;
            }
        }
    }
    return total_size;
}



bool InitializeDataStorage()
{
    if (INIT_DATA_FOLDER_TREE) {
        SPDLOG_INFO("Data folder tree already initialized.");
        return true;
    }

    // Construct the full paths using the root and subdirectory names
    std::vector<std::string> directories = {
        ROOT_DATA_FOLDER,
        IMAGES_FOLDER,
        TELEMETRY_FOLDER,
        EXPERIMENTS_FOLDER,
        LOGGING_FOLDER
    };

    bool success = true;
    for (const auto& dir : directories) {
        if (!make_new_directory(dir)) {
            SPDLOG_CRITICAL("Failed to create directory: {}", dir);
            success = false;
        }
    }

    INIT_DATA_FOLDER_TREE = success;

    if (success) {
        SPDLOG_INFO("Data folder tree initialized successfully.");
    }

    return success;
}

}