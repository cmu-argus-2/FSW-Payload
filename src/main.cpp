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

bool is_fifo(const char *path)
{
    std::error_code ec;
    if (!std::filesystem::is_fifo(path, ec)) {
        if (ec) std::cerr << ec.message() << std::endl;
        return false;
    }
    return true;
}

void ParseCommand(Payload& payload, const std::string& command)
{
    uint8_t cmd_id;
    std::vector<uint8_t> data;
    
    std::istringstream stream(command);
    std::string token;

    // Extract the command ID (first token)
    int cmd_int;
    if (stream >> cmd_int) {
        if (cmd_int < 0 || cmd_int > 255) {
            SPDLOG_ERROR("Command ID out of uint8_t range: {}", cmd_int);
            return;
        }
        cmd_id = static_cast<uint8_t>(cmd_int);
    } else {
        SPDLOG_ERROR("Invalid command format. Could not parse command ID.");
        return;
    }

    // Parse remaining tokens as uint8_t arguments for data
    while (stream >> token) {
        int arg;
        bool is_numeric = true;

        // Check if token is numeric
        for (char c : token) {
            if (!isdigit(c) && !(c == '-' && &c == &token[0])) { // Allow leading negative sign
                is_numeric = false;
                break;
            }
        }

        if (!is_numeric) {
            SPDLOG_ERROR("Invalid argument: '{}'. Not a numeric value.", token);
            return;
        }

        // Convert to integer and check range
        arg = std::stoi(token);
        if (arg < 0 || arg > 255) {
            SPDLOG_ERROR("Argument '{}' out of uint8_t range", token);
            return;
        }

        data.push_back(static_cast<uint8_t>(arg));
    }

    // Logging the parsed command ID and data for verification
    // SPDLOG_INFO("Command ID: {}", static_cast<int>(cmd_id));
    // SPDLOG_INFO("Arguments:");
    for (uint8_t byte : data) {
        SPDLOG_INFO(" {}", static_cast<int>(byte));
    }

    // Send the command to the payload
    payload.AddCommand(cmd_id, data);
}

int main(int argc, char** argv)
{

    SetupLogger();
    // spdlog::set_level(spdlog::level::warn);

    std::string config_path;
    if (argc > 1) {
        config_path = argv[1];
    } else {
        config_path = "config/config.toml";
    }
    SPDLOG_INFO("Using configuration file at {}", config_path);
    
    Configuration config;
    config.LoadConfiguration(config_path);

    Payload& payload = Payload::GetInstance(config);
    payload.Initialize();


    // Initialize pipe for command line interface
    bool read_from_pipe = false;
    std::ifstream pipe;

    const char* fifo_path = IPC_FIFO; // Use predefined FIFO path
    // Check if IPC_FIFO is a FIFO and open it directly
    if (is_fifo(fifo_path)) 
    {
        pipe.open(fifo_path);
        if (pipe.is_open()) 
        {
            read_from_pipe = true;
        } 
        else 
        {
            SPDLOG_WARN("Error: Could not open FIFO {}. Disabling pipe reading.", fifo_path);
        }
    } 
    else 
    {
        SPDLOG_WARN("Error: {} is not a FIFO / named pipe. Disabling pipe reading.", fifo_path);
    }



    std::thread run_thread(&Payload::Run, &payload);

    if (read_from_pipe) 
    {
        std::string command;
        while (std::getline(pipe, command) && payload.IsRunning()) 
        {
            ParseCommand(payload, command);
        }
    }
    SPDLOG_INFO("Exited pipe loop");

    if (payload.IsRunning()) 
    {
        SPDLOG_INFO("Sending shutdown command since payload still running...");
        std::vector<uint8_t> no_data = {};
        // Send shutdown command
        payload.AddCommand(CommandID::SHUTDOWN, no_data);
    }


    // For testing purpsoes 
    /*std::vector<uint8_t> data = {0x01, 0x02, 0x03};
    std::vector<uint8_t> no_data = {};
    payload.AddCommand(CommandID::REQUEST_STATE, no_data);
    payload.AddCommand(CommandID::DEBUG_DISPLAY_CAMERA, no_data);

    // payload.GetRxQueue().PrintAllTasks();
    // payload.AddCommand(CommandID::TURN_OFF_CAMERAS, no_data);
    
    std::thread run_thread(&Payload::Run, &payload);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    payload.AddCommand(CommandID::REQUEST_STATE, no_data);
    payload.AddCommand(CommandID::REQUEST_STATE, no_data);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    // Send shutdown command
    payload.AddCommand(CommandID::SHUTDOWN, no_data);*/


    // Wait for the thread to finish
    run_thread.join();


    return 0;
}