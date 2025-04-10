#include "communication/uart.hpp"

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>   // File control options
#include <errno.h>   // Error number definitions

#include "payload.hpp"
#include "core/errors.hpp"

#include "spdlog/spdlog.h"


static std::array<uint8_t, Packet::INCOMING_PCKT_SIZE> READ_BUF = {0};
static std::array<uint8_t, Packet::OUTGOING_PCKT_SIZE> WRITE_BUF = {0};


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
            LogError(EC::UART_OPEN_FAILED);
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
        LogError(EC::UART_OPEN_FAILED_AFTER_RETRY);
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

    for (int i = 0; i < 3; i++)
    {
        SPDLOG_INFO("Configuring UART port... Attempt {}/3", i+1);
        
        if(tcgetattr(serial_port_fd, &tty) != 0) // get current attributes
        {
            SPDLOG_ERROR("Error from tcgetattr: {}", strerror(errno));
            LogError(EC::UART_GETATTR_FAILED);
            continue; // retry
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

        if (tcsetattr(serial_port_fd, TCSANOW, &tty)) // apply changes
        {
            SPDLOG_ERROR("Error from tcsetattr: {}", strerror(errno));
            LogError(EC::UART_SETATTR_FAILED);
            continue; // retry
        }

        SPDLOG_INFO("UART port configured successfully.");
        return; // success
    }


}

void UART::ClearUpLink()
{
    // Flush data received but not read
    tcflush(serial_port_fd, TCIFLUSH);
    // Flush data written but not transmitted
    tcflush(serial_port_fd, TCOFLUSH);
}

bool UART::FillWriteBuffer(const std::vector<uint8_t>& data)
{
    if (data.size() > Packet::OUTGOING_PCKT_SIZE)
    {
        SPDLOG_ERROR("Data size exceeds outgoing packet size.");
        LogError(EC::UART_WRITE_BUFFER_OVERFLOW);
        return false;
    }
    // Clear the buffer
    std::fill(WRITE_BUF.begin(), WRITE_BUF.end(), 0);
    // For now, just copy the data. TODO: move to buffer once packet data move to std arrays
    std::copy(data.begin(), data.end(), WRITE_BUF.begin());
    return true;
}


bool UART::Send(const std::vector<uint8_t>& data)
{
    if (!port_opened)
    {
        SPDLOG_ERROR("UART port is not open, cannot send data.");
        LogError(EC::UART_NOT_OPEN);
        return false;
    }

    if (!FillWriteBuffer(data))
    {
        return false;
    }

    ssize_t bytes_written = write(serial_port_fd, WRITE_BUF.data(), Packet::OUTGOING_PCKT_SIZE);

    if (bytes_written == -1 || bytes_written != Packet::OUTGOING_PCKT_SIZE) 
    {
        SPDLOG_ERROR("Failed to write to UART port {}: {}", PORT_NAME, strerror(errno));
        LogError(EC::UART_FAILED_WRITE);
        return false;
    }

    SPDLOG_INFO("Sent data of size {} !", bytes_written);
    return true;
}

bool UART::Receive(uint8_t& cmd_id, std::vector<uint8_t>& data)
{
    // Non-blocking 
    ssize_t bytes_read = read(serial_port_fd, READ_BUF.data(), Packet::INCOMING_PCKT_SIZE);

    if (bytes_read <= 0) // No data or error
    {
        return false;
    } 

    if (bytes_read < Packet::INCOMING_PCKT_SIZE) 
    {
        LogError(EC::UART_INCOMPLETE_READ);
        return false;
    }

    cmd_id = READ_BUF[0];
    data.assign(READ_BUF.begin() + 1, READ_BUF.begin() + bytes_read);
    return true;
}


void UART::RunLoop()
{
    _running_loop.store(true);


    if (_connected)
    {
        Connect();
    }

    while (_running_loop.load() && _connected)
    {
        // Busy waiting this thread for now
        
        uint8_t cmd_id;
        std::vector<uint8_t> data;
        Receive(cmd_id, data);
        if (!data.empty())
        {
            sys::payload().AddCommand(cmd_id, data);
        }

        // transmit if there is data to send
        if (!sys::payload().GetTxQueue().IsEmpty())
        {
            std::shared_ptr<Message> msg = sys::payload().GetTxQueue().GetNextMsg();
            bool succ = Send(msg->packet);
            if (succ)
            {
                SPDLOG_INFO("Transmitted message with ID: {}", msg->id);
            } 
            else
            {
                // Need some retry mechanism and fault handling
                SPDLOG_WARN("Failed to transmit message with ID: {}", msg->id);
            }
        }
    }

}

void UART::StopLoop()
{
    _running_loop.store(false);
    if (serial_port_fd != -1)
    {
        close(serial_port_fd);
    }
}