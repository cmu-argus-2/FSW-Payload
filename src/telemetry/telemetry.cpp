#include <cstdio>
#include "unistd.h"
#include "spdlog/spdlog.h"

#include "telemetry/telemetry.hpp"

int CountActiveThreads()
{
    char buffer[128];
    FILE* fp = popen(("ls -1 /proc/" + std::to_string(getpid()) + "/task | wc -l").c_str(), "r");

    if (!fp)
    {
        spdlog::error("Couldn't run active thread count command line");
        return -1; 
    }

    fgets(buffer, sizeof(buffer), fp);
    pclose(fp);

    return std::atoi(buffer);
}

