/*

Main Entry Point for the Payload Flight Software.

Author: Ibrahima Sory Sow

*/
#include <thread>
#include <vector>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <unistd.h>
#include "spdlog/spdlog.h"
#include "payload.hpp"
#include "configuration.hpp"
#include "communication/named_pipe.hpp"
#include "communication/uart.hpp"
#include "core/timing.hpp"
#include "logger.hpp"

int main(int argc, char** argv)
{

    // Set the reference time
    timing::InitializeBootTime();
    
    if (argc > 2) {
        std::string level = argv[2];
        LoggerManager::SetLogSeverity(level);
    }
    else {
        // if no argument is provided then default to info
        LoggerManager::SetLogSeverity("info"); 
    }
    LoggerManager::SetupLogger();


    ////// LOADING CONFIGURATION
    std::string config_path = "config/config.toml";
    // check if path exists
    if (!std::filesystem::exists(config_path)) 
    {
        SPDLOG_ERROR("Configuration file not found at {}", config_path);
        return 1;
    } //TODO back up default configuration (without file if not found)

    SPDLOG_INFO("Using configuration file at {}", config_path);
    std::unique_ptr<Configuration> config = std::make_unique<Configuration>();
    config->LoadConfiguration(config_path);



    ////// ESTABLISHING COMMUNICATION INTERFACE

    std::unique_ptr<Communication> comms_interface;
    // First argument is communication interface, either "UART" or "CLI"
    if (argc > 1) 
    {
        std::string choice_interface = argv[1];
        if (choice_interface == "UART") 
        {
            SPDLOG_INFO("Using UART communication interface");
            // comms_interface = std::make_unique<UARTCommunication>();
        } 
        else if (choice_interface == "CLI") 
        {
            SPDLOG_INFO("Using CLI / NamedPipe communication interface");
            comms_interface = std::make_unique<NamedPipe>();
        } 
        else 
        {
            SPDLOG_ERROR("Invalid communication interface: {}. Use 'UART' or 'CLI'.", choice_interface);
            return 1;
        }
        
    } 
    else // Default to UART
    {
        SPDLOG_INFO("Using UART communication interface");
        // comms_interface = std::make_unique<UARTCommunication>();
    }
    
    Payload& payload = Payload::CreateInstance(std::move(config), std::move(comms_interface));
    payload.Initialize();
    payload.Run(); // Starts the main loop


    return 0;
}