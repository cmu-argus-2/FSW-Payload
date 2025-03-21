#include "communication/uart.hpp"

#include <fcntl.h>   // File control options
#include <errno.h>   // Error number definitions

#include "core/errors.hpp"

#include "spdlog/spdlog.h"


static constexpr speed_t BAUD_RATE = B115200;
static constexpr const char* PORT_NAME = "/dev/ttyTHS0"; // TODO: CONFIG File


UART::UART()
{
}


void UART::OpenPort()
{
    
    for (int i = 0; i < 3; i++)
    {
        serial_port_fd = open(PORT_NAME, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (serial_port_fd == -1) 
        {
            SPDLOG_ERROR("Failed to open UART port {}: {}", PORT_NAME, strerror(errno));
            log_error(EC::UART_OPEN_FAILED);
            failed_open_counter++;
        }
        else
        {
            port_opened = true;
            break;
        }
    }

    if (!port_opened)
    {
        SPDLOG_ERROR("Failed to open UART port after 3 attempts. Exiting.");
        log_error(EC::UART_OPEN_FAILED_AFTER_RETRY);
        // exit(1); // TODO: Can afford to do this when the superloop is implemented
    }
    
}


bool UART::Connect()
{
    if (_connected)
    {
        SPDLOG_WARN("UART already connected.");
        return true;
    }

    OpenPort();
    ConfigurePort();

    _connected = true;
    return _connected;
}

void UART::ConfigurePort()
{

    if(tcgetattr(serial_port_fd, &tty) != 0) // get current attributes
    {
        SPDLOG_ERROR("Error from tcgetattr: {}", strerror(errno));
        log_error(EC::UART_GETATTR_FAILED);
    }

    ClearUpLink();

    // good explanation https://blog.mbedded.ninja/programming/operating-systems/linux/linux-serial-ports-using-c-cpp/

    // set baud rate
    cfsetospeed(&tty, BAUD_RATE); // write speed
    cfsetispeed(&tty, BAUD_RATE); // read speed

    // set parity
    tty.c_cflag &= ~PARENB; // no parity

    // set stop bits
    tty.c_cflag &= ~CSTOPB; // 1 stop bit

    // set data bits
    tty.c_cflag &= ~CSIZE; // clear data size
    tty.c_cflag |= CS8; // 8 bits per byte

    // set control flags
    tty.c_cflag |= ~CRTSCTS; // no hardware flow control
    tty.c_cflag |= CREAD | CLOCAL; // enable receiver, ignore modem control lines

    // set local moes
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // non-canonical mode (raw data)


    // set input flags
    tty.c_iflag &= ~(IXON | IXOFF | IXANY); // disable software flow control
    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL); // Disable any special handling of received bytes

    // set output flags
    tty.c_oflag &= ~(OPOST | ONLCR); // raw output

    // VMIN = 0, VTIME > 0: Read returns when timeout occurs or data is available.
    // VMIN > 0, VTIME = 0: Read blocks until VMIN bytes received.
    // VMIN > 0, VTIME > 0: Read blocks until VMIN bytes or timeout after first byte.
    // VMIN = 0, VTIME = 0: Non-blocking read.
    tty.c_cc[VTIME] = 10; // timeout in deciseconds
    tty.c_cc[VMIN] = 0;

}

void UART::ClearUpLink()
{
    // Flush data received but not read
    tcflush(serial_port_fd, TCIFLUSH);
    // Flush data written but not transmitted
    tcflush(serial_port_fd, TCOFLUSH);
}

void UART::Send(const std::vector<uint8_t>& data)
{

}

void UART::Receive(uint8_t& cmd_id, std::vector<uint8_t>& data)
{

}


void UART::RunLoop()
{

}

void UART::StopLoop()
{
    _running_loop.store(false);
}