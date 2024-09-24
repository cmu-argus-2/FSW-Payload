#ifndef MONITORING_HPP
#define MONITORING_HPP

#include <string_view>


// leveraging tegrastats to monitor the system
void StartTegrastats(std::string_view log_file, int interval = 1000);
void StopTegrastats();



size_t GetFileSize(std::string_view file_path);
size_t GetDirectorySize(std::string_view directory_path);










#endif // MONITORING_HPP