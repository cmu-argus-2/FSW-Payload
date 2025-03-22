#ifndef COMMS_HPP
#define COMMS_HPP

#include <vector>
#include <cstdint>
#include <atomic>

// Asymmetric sizes for send and receive buffers
static constexpr uint8_t OUTGOING_PCKT_SIZE = 250;
static constexpr uint8_t INCOMING_PCKT_SIZE = 32;


// Abstract class for communication interfaces
class Communication
{

public:

    Communication()
        : _connected(false), _running_loop(false) {}
    
    virtual ~Communication() = default; // Ensure proper cleanup for derived classes


    virtual bool Connect() = 0;
    virtual void Disconnect() = 0;
    virtual bool Receive(uint8_t& cmd_id, std::vector<uint8_t>& data) = 0;
    virtual bool Send(const std::vector<uint8_t>& data) = 0;
    virtual void RunLoop() = 0;
    virtual void StopLoop() = 0;
    virtual bool IsConnected() const { return _connected; }
    

protected:

    bool _connected;
    std::atomic<bool> _running_loop;

};


#endif // COMMS_HPP