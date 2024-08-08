/*

Main Entry Point for the Payload Flight Software.

Author: Ibrahima Sory Sow

*/

#include "spdlog/spdlog.h"
#include "payload.hpp"
#include <vector>


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
    
    
    
    Payload payload;
    payload.Initialize();


    // For testing purpsoes 
    int cmd_id = 0;
    std::vector<uint8_t> data = {0x01, 0x02, 0x03};
    std::vector<uint8_t> no_data = {};
    payload.AddCommand(0, data);
    payload.AddCommand(1, data);
    payload.AddCommand(2, no_data);
    payload.AddCommand(3, no_data);

    payload.GetRxQueue().PrintAllTasks();

    payload.Run();


    return 0;
}