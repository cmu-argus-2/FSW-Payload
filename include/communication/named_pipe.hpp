#ifndef NAMED_PIPE_HPP
#define NAMED_PIPE_HPP

#ifndef IPC_FIFO_PATH
#define IPC_FIFO_PATH "/tmp/payload_fifo"
#endif


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
    bool Send(const std::vector<uint8_t>& data) override;
    void RunLoop(Payload* payload) override;
    void StopLoop() override;


private:

    int pipe_fd;
    struct pollfd pfd;

    bool ParseCommand(const std::string& command, uint8_t& cmd_id, std::vector<uint8_t>& data);

};










#endif // NAMED_PIPE_HPP