#ifndef MONITORING_HPP
#define MONITORING_HPP

#include <string>

// leveraging tegrastats to monitor the system
void StartTegrastats(const std::string& log_file, int interval = 1000);
void StopTegrastats();

















#endif // MONITORING_HPP