#include "commands.hpp"
#include "payload.hpp"
#include <iostream>







void request_state(Payload* payload, std::vector<uint8_t> data)
{
    payload->GetState();
    std::cout << "State is: " << ToString(payload->GetState()) << std::endl;

}