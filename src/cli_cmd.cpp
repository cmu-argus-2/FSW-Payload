#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <readline/readline.h>
#include <readline/history.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>



bool is_fifo(const char *path)
{
    std::error_code ec;
    if (!std::filesystem::is_fifo(path, ec)) {
        if (ec) std::cerr << ec.message() << std::endl;
        return false;
    }
    return true;
}


int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    const char* fifo_path = IPC_FIFO; // Use predefined FIFO path


    // Create the FIFO if it doesn't exist
    if (mkfifo(fifo_path, 0666) == -1) {
        if (errno != EEXIST) 
        { // Ignore error if FIFO already exists
            std::cerr << "Error creating FIFO: " << strerror(errno) << std::endl;
            return 1;
        }
    }



    // Check if IPC_FIFO is a FIFO
    if (!is_fifo(fifo_path)) {
        std::cerr << "Error: " << fifo_path << " is not a FIFO / named pipe." << std::endl;
        return 1;
    }

    std::ofstream pipe(fifo_path);
    if (!pipe.is_open()) 
    {
        std::cerr << "Error: Could not open FIFO " << fifo_path << std::endl;
        return 1;
    }

    char* input;

    while ((input = readline("PAYLOAD> ")) != nullptr) 
    {
        if (*input) { add_history(input);  }
            
        std::string cmd(input);  
        free(input);             // Executed regardless of whether input is empty or not
        pipe << cmd << std::endl; // Send command to FSW via the pipe
        pipe.flush(); // Ensure data is written immediately

    }

    return 0;
}