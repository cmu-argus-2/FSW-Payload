/*
A Fast C++ logging library

Attributes
----------
    CRITICAL: int
    ERROR: int
    WARN: int
    INFO: int
    DEBUG: int
    TRACE: int
    OFF: int
*/

#include "logger.hpp"
#include "spdlog/spdlog.h"
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/async.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <map>

#define LOGGER_MAX_FILE_SIZE    5242880 // 5MB
#define LOGGER_MAX_FILES        5

spdlog::level::level_enum LoggerManager::requested_level = spdlog::level::info; 
  
// Terminal prompt to allow the user to choose a desired level of severity
void LoggerManager::PromptLogSeverity()
{
    spdlog::init_thread_pool(8192, 1);
    std::cout << R"(
Choose a log severity level:
    [0] trace       -- Most verbose, useful for debugging low-level details
    [1] debug       -- Debugging messages
    [2] info        -- General information about application's operation
    [3] warn        -- For potential issues
    [4] error       -- For any errors
    [5] critical    -- For any serious errors
    [6] off         -- Disable all the logging

Enter the number corresponding to the desired level:)";

    // Parse input and save it, will be set in SetupLogger
    std::string input;
    std::getline(std::cin, input);
    requested_level = GetLogSeverity(input);
          
}
       
// Method to set up the logger - set the pattern, set the severity level, set up files 
void LoggerManager::SetupLogger(const std::string& file_name){
    
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
    std::string file_path = "./logs/" + file_name; 
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
    logger->set_level(requested_level);
    logger->flush_on(spdlog::level::warn);
}
    
// Method to parse the user input after prompting what severity level they want
spdlog::level::level_enum LoggerManager::GetLogSeverity(const std::string& input) {
    // Map severity to the number / string
    static std::map<std::string, spdlog::level::level_enum> level_map = {
        {"0", spdlog::level::trace},
        {"1", spdlog::level::debug},
        {"2", spdlog::level::info},
        {"3", spdlog::level::warn},
        {"4", spdlog::level::err},
        {"5", spdlog::level::critical},
        {"6", spdlog::level::off},
        {"trace", spdlog::level::trace},
        {"debug", spdlog::level::debug},
        {"info",  spdlog::level::info},
        {"warn",  spdlog::level::warn},
        {"error", spdlog::level::err},
        {"critical", spdlog::level::critical},
        {"off", spdlog::level::off}
    };

    // Convert all characters to lowercase
    std::string input_lowercase = input;
    std::transform(input_lowercase.begin(), input_lowercase.end(), input_lowercase.begin(), ::tolower);

    // Map the string to the corresponding level enum, default is info if not found
    auto it = level_map.find(input_lowercase);
    if (it == level_map.end()) {
        std::cout << "Invalid input, defaulting to INFO\n";
    }
    return it != level_map.end() ? it->second : spdlog::level::info;

}