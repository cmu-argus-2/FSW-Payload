/*

Main Entry Point for the Payload Flight Software.

Author: Ibrahima Sory Sow

*/

#include "spdlog/spdlog.h"
#include "payload.hpp"

int main(int argc, char** argv)
{
    Payload payload;
    payload.Run();


    // For testing purpsoes 
    int cmd_id = 5;
    std::vector<uint8_t> data = {0x01, 0x02, 0x03};
    payload.AddCommandToQueue(static_cast<CommandID>(cmd_id), data);

    spdlog::info("Hello, {}!", "World");

    return 0;
}