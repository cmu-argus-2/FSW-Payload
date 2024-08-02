#ifndef PAYLOAD_HPP
#define PAYLOAD_HPP

#include "queues.hpp"

enum class PayloadState : uint8_t {
    STARTUP = 0x00,
    IDLE = 0x01,
    NOMINAL = 0x02,
    SAFE_MODE = 0x03
};

// Function to convert PayloadState enum to string
const char* ToString(PayloadState state);

class Payload
{
public:
    Payload();
    // ~Payload();
    void run();



private:

    PayloadState state;
    RX_Queue rx_queue;
    TX_Queue tx_queue;
    
    
};

#endif // PAYLOAD_HPP
