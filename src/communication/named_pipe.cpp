#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include "communication/named_pipe.hpp"
#include "payload.hpp"
#include "core/timing.hpp"

// Buffers
static std::array<uint8_t, Packet::INCOMING_PCKT_SIZE+1> READ_BUF = {0}; // specifically +1 for null termination
static Packet::Out WRITE_BUF = {0};

bool IsFifo(const char *path)
{
    std::error_code ec;
    if (!std::filesystem::is_fifo(path, ec)) 
    {
        if (ec) std::cerr << ec.message() << std::endl;
        return false;
    }
    return true;
}

// Function to set a file descriptor to non-blocking mode
void Set_NonBlocking(int fd) 
{
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) 
    {
        SPDLOG_ERROR("Failed to get flags for file descriptor.");
        return;
    }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) 
    {
        SPDLOG_ERROR("Failed to set file descriptor to non-blocking mode.");
    }
}


bool ReadLineFromPipe(int fd, std::string& line) 
{
    static std::string buffer; // Buffer to accumulate data

    // Read data from the file descriptor
    ssize_t bytes_read = read(fd, READ_BUF.data(), Packet::INCOMING_PCKT_SIZE);
    if (bytes_read <= 0) 
    {
        return false; // No data read or error
    }
    // Null-terminate and append to the buffer
    //chunk[bytes_read] = '\0';
    //buffer.append(chunk);
    READ_BUF[bytes_read] = '\0'; // Null-terminate the read buffer
    buffer.append(reinterpret_cast<char*>(READ_BUF.data()), bytes_read); // Append the read data to the buffer

    // Find the first newline in the buffer
    size_t new_line_pos = buffer.find('\n');
    if (new_line_pos != std::string::npos) 
    {
        // Extract the line up to the newline
        line = buffer.substr(0, new_line_pos);
        // Remove the line from the buffer
        buffer = buffer.substr(new_line_pos + 1);

        return true;
    }
    // No complete line yet
    return false;
}


NamedPipe::NamedPipe()
    : Communication() {}


bool NamedPipe::Connect()
{
    const char* fifo_path_in = IPC_FIFO_PATH_IN; // Payload reads from this fifo, external process is writing into it
    const char* fifo_path_out = IPC_FIFO_PATH_OUT; // Payload writes to this fifo, external process is reading from it

    if (_connected) 
    {
        SPDLOG_WARN("NamedPipe already connected.");
        return true;
    }


    // Create the FIFOs if they don't exist
    if (mkfifo(fifo_path_in, 0666) == -1 && errno != EEXIST)
    {
        SPDLOG_ERROR("Error creating FIFO: {}", strerror(errno));
        return 1;
    }

    if (mkfifo(fifo_path_out, 0666) == -1 && errno != EEXIST)
    {
        SPDLOG_ERROR("Error creating FIFO: {}", strerror(errno));
        return 1;
    }

    // Open the input FIFO for reading
    if (IsFifo(fifo_path_in)) 
    {
        pipe_fd_in = open(fifo_path_in, O_RDONLY | O_NONBLOCK);
        if (pipe_fd_in >= 0) 
        {
            Set_NonBlocking(pipe_fd_in);
            _connected = true;
            SPDLOG_INFO("Connected to FIFO {}", fifo_path_in);
        } 
        else 
        {
            SPDLOG_ERROR("Could not open FIFO {}. Disabling pipe reading.", fifo_path_in);
        }
    }

    // Open the output FIFO for writing (NEW)
    if (IsFifo(fifo_path_out)) 
    {
        pipe_fd_out = open(fifo_path_out, O_RDWR | O_NONBLOCK); // write-only requires at least a reader and we don't want that assumption
        if (pipe_fd_out >= 0) 
        {
            SPDLOG_INFO("Connected to FIFO {}", fifo_path_out);
        } 
        else 
        {
            SPDLOG_ERROR("Could not open FIFO {}. Disabling pipe writing.", fifo_path_out);
        }
    }

    return _connected;
}   


void NamedPipe::Disconnect()
{
    if (pipe_fd_in >= 0) 
    {
        close(pipe_fd_in);
        pipe_fd_in = -1;
    }
    if (pipe_fd_out >= 0) 
    {
        close(pipe_fd_out);
        pipe_fd_out = -1;
    }
    _connected = false;
    SPDLOG_WARN("Disconnected from FIFOs");
}


bool NamedPipe::Receive(uint8_t& cmd_id, std::vector<uint8_t>& data) {
    std::string command;
    timing::SleepMs(50); // Obviously not the best way to do this and limits the data rate (also TX)
    // Despite the named pipe setup, we should read a full 'In' packet.
    bool line_received = ReadLineFromPipe(pipe_fd_in, command); // Use custom getline
    // SPDLOG_INFO("Received command?: {}", LineReceived);

    if (line_received) 
    {
        bool status = ParseCommand(command, cmd_id, data);
        return status;
    } 
    else 
    {
        // SPDLOG_WARN("No line received or error reading from pipe.");
        return false;
    }
}




bool NamedPipe::Send(const Packet::Out& data)
{
    if (pipe_fd_out < 0) 
    {
        SPDLOG_ERROR("Write FIFO is not open, cannot send data.");
        return false;
    }
    
    // Convert data to a space-separated string
    std::ostringstream oss;
    for (uint8_t byte : data) 
    {
        oss << static_cast<int>(byte) << " ";
    }
    //oss << "\n";  // append newline for correct `readline()` behavior

    // write to FIFO
    ssize_t bytes_written = write(pipe_fd_out, oss.str().c_str(), oss.str().size());

    if (bytes_written == -1) 
    {
        SPDLOG_ERROR("Failed to write to FIFO {}: {}", IPC_FIFO_PATH_OUT, strerror(errno));
        return false;
    }

    SPDLOG_INFO("Sent data: {}", oss.str());
    return true;
}



void NamedPipe::RunLoop()
{
    _running_loop = true;


    struct pollfd pfds[2];
    pfds[0].fd = pipe_fd_in;  // Read FIFO
    pfds[0].events = POLLIN;  // Wait for data to read

    pfds[1].fd = pipe_fd_out; // Write FIFO
    pfds[1].events = POLLOUT; // Wait for availability to write


    if (!_connected) 
    {
        Connect();
    }

    while (_running_loop && _connected)
    {
        // SPDLOG_INFO("NamedPipe loop running"); 
        // TODO: avoid busy waiting here 
        
        int ret = poll(pfds, 2, 1000); // wait up to 1000ms for events

        if (ret > 0)
        {
            // if somethign to read
            if (pfds[0].revents & POLLIN) 
            {
                uint8_t cmd_id;
                std::vector<uint8_t> data;
                bool recv = Receive(cmd_id, data);
                if (recv)
                {
                    sys::payload().AddCommand(cmd_id, data);
                }
            }

            // check FIFO is ready for writing + TX queue has something for us
            if (pfds[1].revents & POLLOUT && !sys::payload().GetTxQueue().IsEmpty()) 
            {
                std::shared_ptr<Message> msg = sys::payload().GetTxQueue().GetNextMsg();
                bool succ = Send(msg->_packet);
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
        else if (ret == 0) 
        {
            continue; // most likely timeout
        }
        else 
        {
            SPDLOG_ERROR("poll() error: {}", strerror(errno));
        }

    }

}


void NamedPipe::StopLoop()
{
    _running_loop = false;

}


bool NamedPipe::ParseCommand(const std::string& command, uint8_t& cmd_id, std::vector<uint8_t>& data)
{
    std::istringstream stream(command);
    std::string token;

    // Extract the command ID (first token)
    int cmd_int;
    if (stream >> cmd_int) 
    {
        if (cmd_int < 0 || cmd_int > 255) 
        {
            SPDLOG_ERROR("Command ID out of uint8_t range: {}", cmd_int);
            return false;
        }
        cmd_id = static_cast<uint8_t>(cmd_int);
    } else 
    {
        SPDLOG_ERROR("Invalid command format. Could not parse command ID.");
        return false;
    }

    // Parse remaining tokens as uint8_t arguments for data
    while (stream >> token) 
    {
        int arg;
        bool is_numeric = true;

        // Check if token is numeric
        for (char c : token) 
        {
            if (!isdigit(c) && !(c == '-' && &c == &token[0])) 
            { // Allow leading negative sign
                is_numeric = false;
                break;
            }
        }

        if (!is_numeric) 
        {
            SPDLOG_ERROR("Invalid argument: '{}'. Not a numeric value.", token);
            return false;
        }

        // Convert to integer and check range
        arg = std::stoi(token);
        if (arg < 0 || arg > 255) 
        {
            SPDLOG_ERROR("Argument '{}' out of uint8_t range", token);
            return false;
        }

        data.push_back(static_cast<uint8_t>(arg));
    }

    return true;

}