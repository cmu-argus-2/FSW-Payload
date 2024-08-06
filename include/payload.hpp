#ifndef PAYLOAD_HPP
#define PAYLOAD_HPP

#include "spdlog/spdlog.h"

#include "queues.hpp"
#include "commands.hpp"

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


    void Initialize();
    const PayloadState& GetState() const;

    void Run();

    void AddCommand(int command_id, std::vector<uint8_t>& data, int priority = 0);

    const RX_Queue& GetRxQueue() const;
    const TX_Queue& GetTxQueue() const;


private:

    PayloadState state;
    RX_Queue rx_queue;
    TX_Queue tx_queue;


    void SwitchToState(PayloadState new_state);

    void RunStartupHealthProcedures();
    void RetrieveInternalStates();
    
    
};

#endif // PAYLOAD_HPP
