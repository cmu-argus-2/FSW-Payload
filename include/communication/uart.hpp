#ifndef UART_HPP
#define UART_HPP


#include "comms.hpp"

#include <unistd.h>
#include <termios.h> // POSIX terminal control definitions
#include <unistd.h> 
#include <array>



class UART : public Communication
{

public:

    UART();

    bool Connect() override;
    void Disconnect() override;
    bool Receive(uint8_t& cmd_id, std::vector<uint8_t>& data) override;
    bool Send(const Packet::Out& data) override;
    void RunLoop() override;
    void StopLoop() override;

private:

    speed_t BAUD_RATE = B115200;
    const char* PORT_NAME = "/dev/ttyTHS0"; // TODO: CONFIG File
    struct termios tty;
    int serial_port_fd = -1;


    bool port_opened = false;
    int failed_open_counter = 0;

    void OpenPort();
    void ConfigurePort();
    void ClearUpLink(); // flushes data received but not read and data written but not transmitted
    bool FillWriteBuffer(const std::vector<uint8_t>& data);


};

#endif // UART_HPP