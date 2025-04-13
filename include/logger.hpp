#ifndef LOGGER_HPP
#define LOGGER_HPP

#include "spdlog/spdlog.h"

class LoggerManager {
    public:
        static void SetupLogger(const std::string& input);
    private:
        static spdlog::level::level_enum GetLogSeverity(const std::string& input);
};

#endif // LOGGER_HPP