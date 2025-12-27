#include "logger.hpp"
#include "spdlog/spdlog.h"
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/async.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <map>
#include "core/timing.hpp"
#include "core/timing.hpp"
#include <cstdlib>

#define LOGGER_MAX_FILE_SIZE    5242880 // 5MB
#define LOGGER_MAX_FILES        5
       
// Method to set up the logger - set the pattern, set the severity level, set up files 
void LoggerManager::SetupLogger(const std::string& input){
    spdlog::init_thread_pool(5, 1);
    // Set the log pattern

    // %Y-%m-%d %H:%M:%S.%f: timestamp
    // %t: thread id
    // %l: log level
    // %s: source filename
    // %#: source line number
    // %v: the actual text to log
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e][%^%l%$][thread:%t][%s:%#] %v");

    // Creating a color logger
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    // Creating a file rotating logger
    std::string file_path = "./logs/logs_" + std::to_string(timing::GetCurrentTimeMs()) + ".log"; 
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(file_path, LOGGER_MAX_FILE_SIZE, LOGGER_MAX_FILES);
    std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};

    // Make the logger asynchronous
    std::shared_ptr<spdlog::logger> logger;
    logger = std::make_shared<spdlog::async_logger>(
        "logger",
        sinks.begin(), sinks.end(),
        spdlog::thread_pool(),
        spdlog::async_overflow_policy::block
    );

    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
    spdlog::level::level_enum requested_level = GetLogSeverity(input);
    logger->set_level(requested_level);
    logger->flush_on(spdlog::level::warn);
    spdlog::flush_every(std::chrono::seconds(5));
    SPDLOG_INFO("Successfully set up logger. Check logs/ for the recorded files unless specified as OFF");

    std::atexit([]() {
        spdlog::shutdown();
    });
}
    
// Method to parse the argument for logger severity 
spdlog::level::level_enum LoggerManager::GetLogSeverity(const std::string& input) {
    // Convert all characters to uppercase
    std::string input_uppercase = input;
    std::transform(input_uppercase.begin(), input_uppercase.end(), input_uppercase.begin(), ::toupper);
    spdlog::level::level_enum requested_level = spdlog::level::info; 

    // Match the argument to spdlog level enum, default to INFO if invalid
    if (input_uppercase == "TRACE") {
        requested_level = spdlog::level::trace;
    } else if (input_uppercase == "DEBUG") {
        requested_level = spdlog::level::debug;
    } else if (input_uppercase == "INFO") {
        requested_level = spdlog::level::info;
    } else if (input_uppercase == "WARN") {
        requested_level = spdlog::level::warn;
    } else if (input_uppercase == "ERROR") {
        requested_level = spdlog::level::err;
    } else if (input_uppercase == "CRITICAL") {
        requested_level = spdlog::level::critical;
    } else if (input_uppercase == "OFF") {
        requested_level = spdlog::level::off;
    } else {
        SPDLOG_INFO("Invalid argument '{}', defaulting to INFO log severity", input);
        return spdlog::level::info;
    }    

    SPDLOG_INFO("Successfully parsed argument, log severity set to {}", input_uppercase);
    return requested_level;
}