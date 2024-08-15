#ifndef MONITORING_HPP
#define MONITORING_HPP

#include <string>

// leveraging tegrastats to monitor the system
void StartTegrastats(const std::string& log_file, int interval = 1000);
void StopTegrastats();



size_t GetFileSize(const std::string& file_path);
size_t GetDirectorySize(const std::string& directory_path);










#endif // MONITORING_HPP