#include "tegra.hpp"



int main()
{

    TegraTM shared_frame; // Shared memory frame
    RegexContainer regexes; // Regular expression container


    if (!ConfigureSharedMemory(&shared_frame))
    {
        return 1;
    }

    sem_t* sem = InitializeSemaphore(); // Initialize semaphore for shared memory access
    if (!sem)
    {
        return 1;
    }


    RunTegrastatsProcessor(&shared_frame, regexes, sem);

    return 0;
}