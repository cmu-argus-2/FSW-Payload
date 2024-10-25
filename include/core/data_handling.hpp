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





#endif // DATA_HANDLING_HPP