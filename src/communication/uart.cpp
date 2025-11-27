#include "communication/uart.hpp"

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>   // File control options
#include <errno.h>   // Error number definitions

#include "payload.hpp"
#include "core/errors.hpp"
#include "messages.hpp"  // For CRC16 verification

#include "spdlog/spdlog.h"


static Packet::In READ_BUF = {0};
static Packet::Out WRITE_BUF = {0};


UART::UART()
{
}


void UART::OpenPort()
{
    
    for (int i = 0; i < 3; i++)
    {
        serial_port_fd = open(PORT_NAME, O_RDWR | O_NOCTTY | O_NONBLOCK);

        SPDLOG_INFO("***********File descriptor openport: {}", serial_port_fd);
        if (serial_port_fd == -1) 
        {
            SPDLOG_ERROR("Failed to open UART port {}: {}", PORT_NAME, strerror(errno));
            LogError(EC::UART_OPEN_FAILED);
            failed_open_counter++;
        }
        else
        {
            SPDLOG_INFO("UART port {} opened successfully.", PORT_NAME);
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

    if (port_opened) {
        _connected = true;
    }
    SPDLOG_INFO("***********File descriptor uart connect {}", serial_port_fd);

    return _connected;
}

void UART::ConfigurePort()
{
    SPDLOG_INFO("***********File descriptor xconfigure port 1: {}", serial_port_fd);


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
        tty.c_cflag &= ~CRTSCTS; // no hardware flow control
        tty.c_cflag |= CREAD | CLOCAL; // enable receiver, ignore modem control lines

        // set local moes
        // tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // non-canonical mode (raw data)
        tty.c_lflag = 0;

      


        // set input flags
        tty.c_iflag &= ~(IXON | IXOFF | IXANY); // disable software flow control
        tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL); // Disable any special handling of received bytes

        // set output flags
        // tty.c_oflag &= ~(OPOST | ONLCR); // raw output
        tty.c_oflag = 0;

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


bool UART::Send(const Packet::Out& data)
{
    ssize_t bytes_written = write(serial_port_fd, data.data(), Packet::OUTGOING_PCKT_SIZE);

    if (bytes_written == -1) 
    {
        SPDLOG_ERROR("Failed to write to UART port {}: {}", PORT_NAME, strerror(errno));
        LogError(EC::UART_FAILED_WRITE);
        return false;
    }

    SPDLOG_INFO("Sent data of size {} !", bytes_written);
    return true;
}

bool UART::Send(const std::vector<uint8_t>& data)
{
    if (data.empty())
    {
        SPDLOG_ERROR("Cannot send empty packet.");
        return false;
    }

    SPDLOG_INFO("[SEND DEBUG] About to write {} bytes to UART", data.size());
    if (data.size() >= 20) {
        std::string hex_str;
        for (int i = 0; i < 20; i++) {
            char buf[4];
            snprintf(buf, sizeof(buf), "%02x ", data[i]);
            hex_str += buf;
        }
        SPDLOG_INFO("[SEND DEBUG] First 20 bytes being sent: {}", hex_str);
    }

    ssize_t bytes_written = write(serial_port_fd, data.data(), data.size());

    if (bytes_written == -1) 
    {
        SPDLOG_ERROR("Failed to write to UART port {}: {}", PORT_NAME, strerror(errno));
        LogError(EC::UART_FAILED_WRITE);
        return false;
    }

    SPDLOG_INFO("Sent packet of {} bytes (wrote {} bytes)", data.size(), bytes_written);
    
    if (bytes_written != static_cast<ssize_t>(data.size())) {
        SPDLOG_WARN("Partial write! Expected {} bytes, only wrote {}", data.size(), bytes_written);
    }
    
    return true;
}bool UART::Receive(uint8_t& cmd_id, std::vector<uint8_t>& data)
{
    // Non-blocking 
    ssize_t bytes_read = read(serial_port_fd, READ_BUF.data(), Packet::INCOMING_PCKT_SIZE);

    if (bytes_read < 0) // Error
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK)
        {
            SPDLOG_ERROR("UART read error: {}", strerror(errno));
        }
        return false;
    }
    
    if (bytes_read == 0) // No data available
    {
        return false;
    } 

    // if (bytes_read < Packet::INCOMING_PCKT_SIZE) 
    // {
    //     LogError(EC::UART_INCOMPLETE_READ);
    //     return false;
    // }

    SPDLOG_DEBUG("UART read {} bytes: cmd_id={:#x}", bytes_read, READ_BUF[0]);
    
    cmd_id = READ_BUF[0];
    data.assign(READ_BUF.begin() + 1, READ_BUF.begin() + bytes_read); // Got rid of readbuf.begin() +1 because of the way I'm reading currently 
    return true;
}


void UART::RunLoop()
{
    _running_loop.store(true);

    if (!_connected)
    {
        SPDLOG_WARN("UART not connected at RunLoop start, attempting connection...");
        Connect();
    }
    
    SPDLOG_INFO("UART RunLoop started. Connected: {}, FD: {}", _connected, serial_port_fd);

    while (_running_loop.load() && _connected)
    {
        // Busy waiting this thread for now
        
        uint8_t cmd_id;
        std::vector<uint8_t> data;
        Receive(cmd_id, data);
        if (!data.empty())
        {
            SPDLOG_INFO("UART received {} bytes, cmd_id: {}", data.size(), cmd_id);
            // Incoming commands from mainboard are small (32 bytes max) and have NO CRC
            // CRC16 is only used for outgoing large packets (246 bytes) from Jetson to mainboard
            sys::payload().AddCommand(cmd_id, data);
        }

        // transmit if there is data to send
        if (!sys::payload().GetTxQueue().IsEmpty())
        {
            std::shared_ptr<Message> msg = sys::payload().GetTxQueue().GetNextMsg();
            
            // DEBUG: Log packet details before sending
            SPDLOG_INFO("[TX DEBUG] Message ID: {}, packet.size(): {}", msg->id, msg->packet.size());
            if (msg->packet.size() >= 20) {
                std::string hex_str;
                for (int i = 0; i < 20; i++) {
                    char buf[4];
                    snprintf(buf, sizeof(buf), "%02x ", msg->packet[i]);
                    hex_str += buf;
                }
                SPDLOG_INFO("[TX DEBUG] First 20 bytes: {}", hex_str);
            }
            
            bool succ = Send(msg->packet);  // Use vector directly for variable-length packets
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

// ADDED DISCONNECT
void UART::Disconnect()
{
    if (!_connected) {
        SPDLOG_WARN("UART is already disconnected.");
        return;
    }

    // Perform any clean-up actions before disconnecting
    if (serial_port_fd != -1) {
        // Close the UART port
        if (close(serial_port_fd) == -1) {
            SPDLOG_ERROR("Failed to close UART port {}: {}", PORT_NAME, strerror(errno));
            LogError(EC::UART_CLOSE_FAILED);
        } else {
            SPDLOG_INFO("UART port {} disconnected successfully.", PORT_NAME);
        }
        serial_port_fd = -1;  // Reset file descriptor
    }

    // Reset the connected state
    _connected = false;

    // Perform any other necessary state resets
    port_opened = false;

    SPDLOG_INFO("UART disconnected.");
}











// // #CHAOS BEYOND

// #include <termios.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <stdio.h>
// #include <string.h>
// #include <errno.h>
// #include <stdint.h>

// #define UART_DEVICE "/dev/ttyTHS1"
// #define UART_BAUDRATE B115200

// // -------------------------------
// // UART Setup
// // -------------------------------
// int uart_setup(const char *device, speed_t baud)
// {
//     int fd = open(device, O_RDWR | O_NOCTTY | O_SYNC);
//     if (fd < 0) {
//         SPDLOG_ERROR("open");
//         return -1;
//     }

//     struct termios tty;
//     if (tcgetattr(fd, &tty) != 0) {
//         SPDLOG_ERROR("tcgetattr");
//         close(fd);
//         return -1;
//     }

//     // Set input/output baud rates
//     cfsetospeed(&tty, baud);
//     cfsetispeed(&tty, baud);

//     // Configure 8N1, no flow control
//     tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;   // 8 data bits
//     tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);  // no parity, 1 stop bit, no hw flow
//     tty.c_cflag |= (CLOCAL | CREAD);               // enable receiver

//     tty.c_lflag = 0;                               // raw input mode
//     tty.c_oflag = 0;                               // raw output
//     tty.c_iflag &= ~(IXON | IXOFF | IXANY);        // no sw flow control

//     tty.c_cc[VMIN]  = 1;                           // block until 1 byte
//     tty.c_cc[VTIME] = 0;                           // no inter-byte timeout

//     if (tcsetattr(fd, TCSANOW, &tty) != 0) {
//         SPDLOG_ERROR("tcsetattr");
//         close(fd);
//         return -1;
//     }

//     tcflush(fd, TCIOFLUSH);
//     return fd;
// }

// // -------------------------------
// // UART Send
// // -------------------------------
// ssize_t uart_send(int fd, const uint8_t *data, size_t len)
// {
//     ssize_t written = write(fd, data, len);
//     if (written < 0) SPDLOG_ERROR("uart_send: write");
//     tcdrain(fd);  // wait for all data to transmit
//     return written;
// }

// // -------------------------------
// // UART Receive (blocking read)
// // -------------------------------
// ssize_t uart_receive(int fd, uint8_t *buf, size_t len)
// {
//     ssize_t n = read(fd, buf, len);
//     if (n < 0) SPDLOG_ERROR("uart_receive: read");
//     return n;
// }

// // -------------------------------
// // UART Close
// // -------------------------------
// void uart_close(int fd)
// {
//     if (fd >= 0)
//         close(fd);
// }

// // // -------------------------------
// // // Example Main
// // // -------------------------------
// // int main(void)
// // {
// //     int fd = uart_setup(UART_DEVICE, UART_BAUDRATE);
// //     if (fd < 0) {
// //         fprintf(stderr, "Failed to open UART\n");
// //         return 1;
// //     }

// //     printf("UART opened successfully on %s\n", UART_DEVICE);

// //     // Example send
// //     const char *msg = "Hello from Jetson!\n";
// //     uart_send(fd, (const uint8_t *)msg, strlen(msg));

// //     // Example receive
// //     uint8_t buf[128];
// //     ssize_t n = uart_receive(fd, buf, sizeof(buf));
// //     if (n > 0) {
// //         printf("Received %zd bytes: ", n);
// //         fwrite(buf, 1, n, stdout);
// //         printf("\n");
// //     }

// // //     uart_close(fd);
// // //     return 0;
// // // }

