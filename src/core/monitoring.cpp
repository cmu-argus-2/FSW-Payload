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
