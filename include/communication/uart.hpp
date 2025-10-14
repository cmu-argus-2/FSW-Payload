
// #ifndef UART_HPP
// #define UART_HPP

// #include <string>
// #include <vector>
// #include <termios.h>
// #include <spdlog/spdlog.h>

// class UART {
// public:
//     // Constructor that takes the UART port name
//     UART(const std::string& port_name);

//     // Connect to the UART port and configure it
//     bool Connect();

//     // Send data over the UART port
//     bool Send(const std::vector<uint8_t>& data);

//     // Receive data from the UART port
//     bool Receive(std::vector<uint8_t>& buffer);

//     // Disconnect and close the UART port
//     void Disconnect();

// private:
//     // Private members to store UART state and file descriptor
//     std::string port_name;
//     int serial_port_fd;
//     bool connected;

//     // Helper function to configure the UART port
//     bool ConfigurePort();
// };

// #endif // UART_HPP


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

    speed_t BAUD_RATE = B57600; // changed from 115200 - change back to 57600
    const char* PORT_NAME = "/dev/ttyTHS1"; // TODO: CONFIG File - was  0 but 0 doesn't exist  in list of devices
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


