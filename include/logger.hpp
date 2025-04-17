#ifndef LOGGER_HPP
#define LOGGER_HPP

#include "spdlog/spdlog.h"

class LoggerManager {
    public:
        static void SetupLogger(const std::string& file_name="");
        static spdlog::level::level_enum requested_level;
        static void SetLogSeverity(const std::string& input);
};

#endif // LOGGER_HPP