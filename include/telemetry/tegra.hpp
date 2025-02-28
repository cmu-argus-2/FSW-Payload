/*
Tegrastats output parser and processor 

Author: Ibrahima S. Sow
*/


#ifndef TEGRA_HPP
#define TEGRA_HPP

#include <unistd.h>
#include <iostream>
#include <sys/stat.h>        /* For mode constants */
#include <semaphore.h>
#include <string>
#include <regex>

// Constants for the shared memory and semaphore paths
#define TEGRASTATS_SHARED_MEM "/tm_tegrastats_shared_mem"
#define TEGRASTATS_SEM "/tm_tegrastats_sem"

// Interval between log outputs to stdout of the tegrastats statistics
#define TEGRASTATS_INTERVAL 200 // Also determines the update rate of the shared memory



struct TegraTM
{
    // Letting the main process converting to percentage

    int ram_used; // Used RAM in MB
    int ram_total; // Total RAM in MB
    int swap_used; // Used swap memory in MB
    int swap_total; // Available swap memory in MB
    int active_cores; // Number of active CPU cores
    int cpu_load[6]; // CPU cores load in %
    int gpu_freq; // GPU frequency in %
    int vdd_in; // VDD_IN in mW
    int vdd_cpu_gpu_cv; // VDD_CPU_GPU_CV in mW 
    int vdd_soc; // VDD_SOC in mW

    float cpu_temp; // CPU temperature in Celsius
    float gpu_temp; // GPU temperature in Celsius

    uint8_t change_flag; // Flag to indicate if the data has changed

    TegraTM();
};

constexpr size_t SHARED_MEM_SIZE = sizeof(TegraTM);

// Extremely expensive to construct regex objects every time
struct RegexContainer 
{   
    // Regular expressions for pattern matching
    const std::regex ram_regex;
    const std::regex swap_regex;
    const std::regex cpu_regex;
    const std::regex cpu_load_regex;
    const std::regex gpu_regex;
    const std::regex cpu_temp_regex;
    const std::regex gpu_temp_regex;
    const std::regex vdd_in_regex;
    const std::regex vdd_cpu_gpu_cv_regex;
    const std::regex vdd_soc_regex;

    std::smatch match;

    RegexContainer();
};


void PrintTegraTM(TegraTM& frame);

void ParseTegrastatsLine(const std::string& line, RegexContainer& regexes, TegraTM& frame);


void StopTegrastats();

/* Writer side */

// Allocate the sized shared memory and map it to the argument struct pointer
bool ConfigureSharedMemory(TegraTM*& shared_mem);

// Create a semaphore for synchronization
sem_t* InitializeSemaphore();

// Run continuously the tegrastats process
void RunTegrastatsProcessor(TegraTM* frame, RegexContainer& regexes, sem_t* sem);

/* Reading side */

// Link the argument struct pointer to the shared memory. Requires the memory to be already configured.
bool LinkToSharedMemory(TegraTM*& shared_mem);

// Link to synchronization semaphore
sem_t* LinkToSemaphore();



#endif // TEGRA_HPP