/*

Main Entry Point for the Payload Flight Software.

Author: Ibrahima Sory Sow

*/
#include <thread>
#include <vector>
#include "spdlog/spdlog.h"
#include <unistd.h>
#include "payload.hpp"
//#include <torch/torch.h>
#include "configuration.hpp"


void SetupLogger()
{
    // Set the log pattern

    // %Y-%m-%d %H:%M:%S.%f: timestamp
    // %t: thread id
    // %l: log level
    // %s: source filename
    // %#: source line number
    // %v: the actual text to log
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e][%^%l%$][thread:%t][%s:%#] %v");
}


int main(int argc, char** argv)
{
    
    SetupLogger();
    // spdlog::set_level(spdlog::level::warn);

    std::string config_path;
    if (argc > 1) {
        config_path = argv[1];
    } else {
        config_path = "configuration/configuration.toml";
    }
    SPDLOG_INFO("Using configuration file at {}", config_path);
    
    Configuration config;
    config.LoadConfiguration(config_path);

    
    Payload payload(config);
    payload.Initialize();

    
    // For testing purpsoes 
    std::vector<uint8_t> data = {0x01, 0x02, 0x03};
    std::vector<uint8_t> no_data = {};


    payload.AddCommand(CommandID::REQUEST_STATE, no_data);
    payload.AddCommand(CommandID::DEBUG_DISPLAY_CAMERA, no_data);

    // payload.GetRxQueue().PrintAllTasks();
    //payload.AddCommand(CommandID::TURN_OFF_CAMERAS, no_data);

    
    
    std::thread run_thread(&Payload::Run, &payload);
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Send shutdown command
    payload.AddCommand(CommandID::SHUTDOWN, no_data);

    // Wait for the thread to finish
    run_thread.join();


    return 0;
}