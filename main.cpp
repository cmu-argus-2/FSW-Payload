/*

Main Entry Point for the Payload Flight Software.

Author: Ibrahima Sory Sow

*/
#include <thread>
#include <vector>
#include "spdlog/spdlog.h"
#include "payload.hpp"


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
    
    
    
    Payload payload;
    payload.Initialize();

    
    // For testing purpsoes 
    int cmd_id = 0;
    std::vector<uint8_t> data = {0x01, 0x02, 0x03};
    std::vector<uint8_t> no_data = {};


    payload.AddCommand(CommandID::REQUEST_STATE, no_data);
    payload.AddCommand(CommandID::DISPLAY_CAMERA, no_data);

    // payload.GetRxQueue().PrintAllTasks();
    
    // For debugging purposes
    std::thread run_thread(&Payload::Run, &payload);

    // wait 20 seconds
    std::this_thread::sleep_for(std::chrono::seconds(5));

    payload.AddCommand(CommandID::REQUEST_STATE, no_data);


    // Send shutdown command
    payload.AddCommand(CommandID::SHUTDOWN, no_data);

    // Wait for the thread to finish
    run_thread.join();


    return 0;
}