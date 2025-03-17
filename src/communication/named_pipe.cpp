#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include "communication/named_pipe.hpp"
#include "payload.hpp"
#include "core/timing.hpp"


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
    char chunk[1024];          // Temporary read buffer

    // Read data from the file descriptor
    ssize_t bytesRead = read(fd, chunk, sizeof(chunk) - 1);
    if (bytesRead <= 0) 
    {
        return false; // No data read or error
    }
    // Null-terminate and append to the buffer
    chunk[bytesRead] = '\0';
    buffer.append(chunk);

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

    // Create the FIFO if it doesn't exist
    if (mkfifo(fifo_path_in, 0666) == -1) {
        if (errno != EEXIST) 
        { // Ignore error if FIFO already exists
            std::cerr << "Error creating FIFO: " << strerror(errno) << std::endl;
            return 1;
        }
    }

    // Check if the path is a FIFO and open it directly
    if (IsFifo(fifo_path_in)) 
    {
        pipe_fd = open(fifo_path_in, O_RDONLY | O_NONBLOCK);
        if (pipe_fd >= 0) 
        {
            Set_NonBlocking(pipe_fd);
            _connected = true;
            SPDLOG_INFO("Connected to FIFO {}", fifo_path_in);
        } 
        else 
        {
            SPDLOG_WARN("Error: Could not open FIFO {}. Disabling pipe reading.", fifo_path_in);
        }
    } 
    else 
    {
        SPDLOG_WARN("Error: {} is not a FIFO / named pipe. Disabling pipe reading.", fifo_path_in);
    }

    return _connected;
}   


void NamedPipe::Disconnect()
{
    if (pipe_fd >= 0) 
    {
        close(pipe_fd);
        pipe_fd = -1;
    }
    _connected = false;
    SPDLOG_WARN("Disconnected from FIFO");
}

/*
bool NamedPipe::Receive(uint8_t& cmd_id, std::vector<uint8_t>& data)
{

    int ret = poll(&pfd, 1, 100); // Wait up to 100 ms
    bool flag = false;
    
    if (ret > 0)
    {
        std::string command;
        std::getline(pipe, command);
        ParseCommand(command, cmd_id, data);
        flag = true;
    } // == 0 means no data to read
    else if (ret < 0)
    {
        SPDLOG_ERROR("Error polling FIFO: {}", strerror(errno));
    }

    return flag;

}
*/


bool NamedPipe::Receive(uint8_t& cmd_id, std::vector<uint8_t>& data) {
    std::string command;
    timing::SleepMs(40); // Obviously not the best way to do this and limits the data rate (also TX)
    bool LineReceived = ReadLineFromPipe(pipe_fd, command); // Use custom getline
    // SPDLOG_INFO("Received command?: {}", LineReceived);

    if (LineReceived) 
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




bool NamedPipe::Send(const std::vector<uint8_t>& data)
{
    // Log data to console - TODO in the meantime
    std::ostringstream oss;
    for (uint8_t byte : data) {
        oss << static_cast<int>(byte) << " ";
    }
    SPDLOG_INFO("Sending data: {}", oss.str());
    return true;
}



void NamedPipe::RunLoop()
{
    _running_loop = true;
    pfd.fd = pipe_fd; // File descriptor for the pipe
    pfd.events = POLLIN; // Monitor for data to read

    if (!_connected) 
    {
        Connect();
    }

    while (_running_loop && _connected)
    {
        // SPDLOG_INFO("NamedPipe loop running"); 
        // TODO: avoid busy waiting here 
        
        // Receive new command
        uint8_t cmd_id;
        std::vector<uint8_t> data;
        bool recv = Receive(cmd_id, data);

        if (recv)
        {
            sys::payload().AddCommand(cmd_id, data);
        } 

        // Transmit messages
        while (!sys::payload().GetTxQueue().IsEmpty())
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