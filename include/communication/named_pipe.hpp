#ifndef NAMED_PIPE_HPP
#define NAMED_PIPE_HPP

#include <string>
#include <fstream>
#include <poll.h>
#include "comms.hpp"


bool IsFifo(const char *path);

class NamedPipe : public Communication // FIFO
{

public:

    NamedPipe();

    bool Connect() override;
    void Disconnect() override;
    bool Receive(uint8_t& cmd_id, std::vector<uint8_t>& data) override;
    bool Send(const Packet::Out& data) override;
    void RunLoop() override;
    void StopLoop() override;


private:

    int pipe_fd_in;
    int pipe_fd_out;
    struct pollfd pfd;

    bool ParseCommand(const std::string& command, uint8_t& cmd_id, std::vector<uint8_t>& data);

};

#endif // NAMED_PIPE_HPP