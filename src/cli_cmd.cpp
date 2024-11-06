#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <readline/readline.h>
#include <readline/history.h>


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
    char* input;

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <pipe_name>" << std::endl;
        return 1;
    }

    if (!is_fifo(argv[1]))
    {
        std::cerr << "Error: " << argv[1] << " is not a FIFO / named pipe." << std::endl;
        return 1;
    }

    std::ofstream pipe(argv[1]);
    if (!pipe.is_open())
    {
        std::cerr << "Error: Could not open FIFO " << argv[1] << std::endl;
        return 1;
    }

    while ((input = readline("FSW> ")) != nullptr) 
    {
        if (*input) { add_history(input);  }
            
        std::string cmd(input);  
        free(input);             // Executed regardless of whether input is empty or not
        pipe << cmd << std::endl; // Send command to FSW via the pipe
        pipe.flush(); // Ensure data is written immediately

    }

    return 0;
}