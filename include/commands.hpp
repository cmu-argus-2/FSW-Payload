#ifndef COMMAND_HPP
#define COMMAND_HPP

#include <cstdint>
#include <cstddef>


enum class CommandID : uint8_t {
    START = 0x10,
    SHUTDOWN = 0x11,
    REQUEST_STATE = 0x12
};

constexpr CommandID ALL_COMMAND_IDS[] = 
{
    CommandID::START,
    CommandID::SHUTDOWN,
    CommandID::REQUEST_STATE
};


constexpr size_t COMMAND_NUMBER = sizeof(ALL_COMMAND_IDS) / sizeof(ALL_COMMAND_IDS[0]);




#endif // COMMAND_HPP