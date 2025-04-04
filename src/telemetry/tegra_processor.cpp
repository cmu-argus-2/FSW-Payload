#include "spdlog/spdlog.h"
#include "telemetry/tegra.hpp"

#define SAMPLE_DATA_FILENAME "src/telemetry/samples/tegrastats_data.txt"

void EmulateTegrastatsProcessor(TegraTM* shared_frame, RegexContainer& regexes, sem_t* sem)
{
    while (true) // high-level infinite loop to emulate continuous data
    {
        FILE* file = fopen(SAMPLE_DATA_FILENAME, "r");
        if (!file) 
        {
            spdlog::error("Failed to open sample data file: {}", SAMPLE_DATA_FILENAME);
            return;
        }

        char line[256];
        TegraTM frame; // Buffer for the parsed data
        frame.change_flag = 1; // Writer set that flag to 1 by default

        while (fgets(line, sizeof(line), file)) 
        {
            // Parsing takes ~0.1ms on Jetson Orin Nano, more than ok for our purposes
            ParseTegrastatsLine(line, regexes, frame);

            sem_wait(sem); 
            SPDLOG_INFO("Reading flag before modification {}", shared_frame->change_flag);
            memcpy(shared_frame, &frame, sizeof(frame));
            SPDLOG_INFO("Setting the reading flag to {}", shared_frame->change_flag);
            SPDLOG_INFO("Data written to shared memory. (e.g {} RAM used, CPU Core 1 load: {}%, ...)", shared_frame->ram_used, shared_frame->cpu_load[0]);
            sem_post(sem);

            

            // Simulate wait to replicate tegrastats behavior
            usleep(TEGRASTATS_INTERVAL * 1000); 
        }

        fclose(file); // Close to reopen in the next iteration
    }

}


int main(int argc, char* argv[])
{

    TegraTM* shared_frame; // Shared memory frame
    RegexContainer regexes; // Regular expression container


    if (!ConfigureSharedMemory(shared_frame))
    {
        return 1;
    }

    sem_t* sem = InitializeSemaphore(); // Initialize semaphore for shared memory access
    if (!sem)
    {
        return 1;
    }


    if (argc > 1)
    {
        if (std::string(argv[1]) == "emulate")
        {
            spdlog::info("Running tegrastats process in emulation mode.");
            EmulateTegrastatsProcessor(shared_frame, regexes, sem);
            return 0;
        }
    }

    spdlog::info("Running tegrastats process on target hardware.");
    RunTegrastatsProcessor(shared_frame, regexes, sem);

    return 0;
}