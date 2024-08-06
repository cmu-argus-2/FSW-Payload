/*

Main Entry Point for the Payload Flight Software.

Author: Ibrahima Sory Sow

*/

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
    
    
    
    Payload payload;
    payload.Initialize();


    // For testing purpsoes 
    int cmd_id = 0;
    std::vector<uint8_t> data = {0x01, 0x02, 0x03};
    payload.AddCommand(cmd_id, data);


    payload.GetRxQueue().PrintAllTasks();




    payload.Run();




    return 0;
}