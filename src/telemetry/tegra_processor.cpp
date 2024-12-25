#include "tegra.hpp"



int main(int argc, char* argv[])
{

    TegraTM shared_frame; // Shared memory frame
    RegexContainer regexes; // Regular expression container
    sem_t* sem; // Semaphore for synchronization

    if (!ConfigureSharedMemory(&shared_frame))
    {
        return 1;
    }

    if (!InitializeSemaphore(sem))
    {
        return 1;
    }


    RunTegrastatsProcessor(&shared_frame, regexes, sem);

    return 0;
}