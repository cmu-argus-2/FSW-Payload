#ifndef COMMS_HPP
#define COMMS_HPP

#include <vector>
#include <cstdint>
#include <core/task.hpp>
#include <queues.hpp>

// Abstract class for communication interfaces
class Communication
{

public:

    Communication(RX_Queue* rx_queue, TX_Queue* tx_queue)
        : rx_queue(rx_queue), tx_queue(tx_queue), connected(false) {}
    
    virtual ~Communication() = default; // Ensure proper cleanup for derived classes


    virtual void Connect() = 0;
    virtual void Disconnect() = 0;
    virtual void Receive(std::vector<uint8_t>& data) = 0;
    virtual void Send(const std::vector<uint8_t>& data) = 0;


private:

    RX_Queue* rx_queue;
    TX_Queue* tx_queue;
    bool connected;

};


#endif // COMMS_HPP