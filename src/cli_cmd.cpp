#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <atomic>
#include <thread>
#include <readline/readline.h>
#include <readline/history.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/select.h>
#include <fcntl.h>
#include <unistd.h>


std::atomic<bool> running(true);

bool IsFifo(const char *path)
{
    std::error_code ec;
    if (!std::filesystem::is_fifo(path, ec)) {
        if (ec) std::cerr << ec.message() << std::endl;
        return false;
    }
    return true;
}

// Background thread function to read responses from FIFO without busy waiting
void ReadResponses(const char* fifo_path_out)
{
    int fd = open(fifo_path_out, O_RDONLY | O_NONBLOCK); // Open FIFO non-blocking
    if (fd == -1) {
        std::cerr << "Error opening FIFO for reading: " << strerror(errno) << std::endl;
        return;
    }

    char buffer[1024];

    while (running.load()) 
    {
        fd_set fds;
        FD_ZERO(&fds); // clears the set of file descriptors
        FD_SET(fd, &fds); // adds the file descriptor to the set

        struct timeval timeout = { .tv_sec = 2, .tv_usec = 0 };  // 2-second timeout

        int ready = select(fd + 1, &fds, nullptr, nullptr, &timeout); // monitor the file descriptor until it is ready for reading

        if (ready > 0 && FD_ISSET(fd, &fds)) 
        {
            ssize_t bytes_read = read(fd, buffer, sizeof(buffer) - 1);
            if (bytes_read > 0) 
            {
                buffer[bytes_read] = '\0';


                // Duplicate the current input line (Fix)
                char* saved_line = strdup(rl_line_buffer);
                int saved_point = rl_point;

                // Clear current line
                printf("\r\033[K");  // ANSI escape code for clear line

                // Print the incoming message
                std::cout << "\033[1;31m[PAYLOAD RESPONSE]:\033[0m " << buffer << std::endl;

                // Restore prompt and input
                printf("PAYLOAD> %s", saved_line);
                fflush(stdout);
                rl_point = saved_point;
                rl_insert_text(saved_line);
                rl_redisplay();
                
                free(saved_line);  // Free the duplicated string

            }
        } else if (ready == -1) 
        {
            std::cerr << "Error in select(): " << strerror(errno) << std::endl;
            break;
        }
    }

    close(fd);
}

static constexpr uint8_t CMD_PCKT_SIZE = 32;

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{   
    uint8_t outgoing_packet_size = CMD_PCKT_SIZE; // Default packet size
    if (argc > 1) 
    {
        try {
            int size = std::stoi(argv[1]);
            if (size > 0 && size <= 32) 
            {
                outgoing_packet_size = static_cast<uint8_t>(size);
            } else {
                std::cerr << "Invalid packet size. Must be between 1 and 255. Using default size: " 
                          << static_cast<int>(CMD_PCKT_SIZE) << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing packet size: " << e.what() 
                      << ". Using default size: " << static_cast<int>(CMD_PCKT_SIZE) << std::endl;
        }
    } else
    {
        std::cerr << "Using default cmd size: " << static_cast<int>(CMD_PCKT_SIZE) << std::endl;
    }


    const char* fifo_path_in = IPC_FIFO_PATH_IN; // Payload reads from this fifo, external process is writing into it
    const char* fifo_path_out = IPC_FIFO_PATH_OUT; // Payload writes to this fifo, external process is reading from it

    // Check if paths are FIFOs
    if (!IsFifo(fifo_path_in)) 
    {
        std::cerr << "Error: " << fifo_path_in << " is not a FIFO / named pipe." << std::endl;
        return 1;
    }
    if (!IsFifo(fifo_path_out)) 
    {
        std::cerr << "Error: " << fifo_path_out << " is not a FIFO / named pipe." << std::endl;
        return 1;
    }


    // Create the FIFOs if they don't exist
    if (mkfifo(fifo_path_in, 0666) == -1 && errno != EEXIST)
    {
        // Ignore error if FIFO already exists
        std::cerr << "Error creating FIFO: " << strerror(errno) << std::endl;
        return 1;
    }

    if (mkfifo(fifo_path_out, 0666) == -1 && errno != EEXIST)
    {
        std::cerr << "Error creating FIFO: " << strerror(errno) << std::endl;
        return 1;
    }



    std::ofstream pipe(fifo_path_in);
    if (!pipe.is_open()) 
    {
        std::cerr << "Error: Could not open FIFO " << fifo_path_in << std::endl;
        return 1;
    }

    std::thread response_reader_thread(ReadResponses, fifo_path_out);

    char* input;
    while ((input = readline("PAYLOAD> ")) != nullptr) 
    {
        if (*input) { add_history(input); }

        std::string cmd(input);
        free(input); // Executed regardless of whether input is empty or not

        // Ensure the command string is of size outgoing_packet_size (assumes we won't overflow)
        if (cmd.size() < outgoing_packet_size && !cmd.empty()) 
        {
            size_t padding_needed = outgoing_packet_size - cmd.size();
            for (size_t i = 0; i < padding_needed*2; ++i) 
            {
            cmd += (i % 2 == 0) ? ' ' : '0'; // Append space and zero alternately
            }
        } 

        if (!cmd.empty()) 
        {
            std::cout << "Sending command: " << cmd << std::endl;
        }

        pipe << cmd << std::endl; // Send command to FSW via the pipe
        pipe.flush(); // Ensure data is written immediately
    }

    running.store(false);
    response_reader_thread.join();

    return 0;
}