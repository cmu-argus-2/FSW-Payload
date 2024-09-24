#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <sys/stat.h>

#include "core/monitoring.hpp"

void StartTegrastats(std::string_view log_file, int interval) 
{
    std::string command = "tegrastats --interval " + std::to_string(interval) + " --logfile " + std::string(log_file) + " &";
    int result = std::system(command.c_str());
    if (result != 0) {
        std::cerr << "Failed to start tegrastats." << std::endl;
    }
}

void StopTegrastats() 
{
    std::string command = "tegrastats --stop"; // Global command, stops any running tegrastats
    int result = std::system(command.c_str());
    if (result != 0) {
        std::cerr << "Failed to stop tegrastats." << std::endl;
    }
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
