#include "telemetry/tegra.hpp"

RegexContainer::RegexContainer()
        : ram_regex(R"(RAM (\d+)/(\d+)MB)"),
            swap_regex(R"(SWAP (\d+)/(\d+)MB)"),
            cpu_regex(R"(CPU \[((?:\d+%@\d+,?|off,?)+)\])"),
            cpu_load_regex(R"((\d+)%@)"),
            gpu_regex(R"(GR3D_FREQ (\d+)%?)"),
            cpu_temp_regex(R"(cpu@([\d.]+)C)"),
            gpu_temp_regex(R"(gpu@([\d.]+)C)"),
            vdd_in_regex(R"(VDD_IN (\d+)mW)"),
            vdd_cpu_gpu_cv_regex(R"(VDD_CPU_GPU_CV (\d+)mW)"),
            vdd_soc_regex(R"(VDD_SOC (\d+)mW)"),
            match()
    {}


void PrintTegraTM(TegraTM& frame)
{
    
    std::cout << "== Tegra Stats ==" << "\n";
    
    std::cout << "RAM Used: " << frame.ram_used << " MB" << "\n";
    std::cout << "RAM Total: " << frame.ram_total << " MB" << "\n";
    std::cout << "Swap Used: " << frame.swap_used << " MB" << "\n";
    std::cout << "Swap Total: " << frame.swap_total << " MB" << "\n";
    std::cout << "Active Cores: " << frame.active_cores << "\n";
    std::cout << "CPU Load: ";
    for (int i = 0; i < 6; ++i) {
        if (frame.cpu_load[i] == -1) {
            std::cout << "off ";
            continue;
        }
        std::cout << frame.cpu_load[i] << "% ";
    }
    std::cout << "\n";
    std::cout << "GPU Frequency: " << frame.gpu_freq << "%" << "\n";
    std::cout << "CPU Temperature: " << frame.cpu_temp << " C" << "\n";
    std::cout << "GPU Temperature: " << frame.gpu_temp << " C" << "\n";
    std::cout << "VDD_IN: " << frame.vdd_in << " mW" << "\n";
    std::cout << "VDD_CPU_GPU_CV: " << frame.vdd_cpu_gpu_cv << " mW" << "\n";
    std::cout << "VDD_SOC: " << frame.vdd_soc << " mW" << std::endl;
}



void ParseTegrastatsLine(const std::string& line, RegexContainer& regexes, TegraTM& frame) 
{
    /*
    Example tegrastats line:
    12-23-2024 22:54:11 RAM 3162/7620MB (lfb 4x2MB) SWAP 0/3810MB (cached 0MB) 
    CPU [1%@729,0%@729,0%@729,0%@729,0%@729,0%@729] GR3D_FREQ 1% cpu@62.843C soc2@61.968C soc0@60.625C gpu@61.687C tj@62.843C soc1@61.687C 
    VDD_IN 3913mW/3922mW VDD_CPU_GPU_CV 598mW/598 VDD_SOC 1238mW/1237mW
    */
    

    // Extract RAM usage
    if (std::regex_search(line, regexes.match, regexes.ram_regex)) 
    {
        frame.ram_used = std::stoi(regexes.match[1].str());
        frame.ram_total = std::stoi(regexes.match[2].str());
    }

    // Extract SWAP usage
    if (std::regex_search(line, regexes.match, regexes.swap_regex)) 
    {
        frame.swap_used = std::stoi(regexes.match[1].str());
        frame.swap_total = std::stoi(regexes.match[2].str());
    }

    // Extract CPU loads
    if (std::regex_search(line, regexes.match, regexes.cpu_regex)) 
    {   
        // need to process something like: 1%@729,0%@729,0%@729,0%@729,0%@729,0%@729
        std::string cpu_load_str = regexes.match[1].str();
        // if some cores are inactive, it can be like: [0%@729,0%@729,0%@729,0%@729,off,off]

        // Regex to match active cores
        std::sregex_iterator iter(cpu_load_str.begin(), cpu_load_str.end(), regexes.cpu_load_regex);
        std::sregex_iterator end;

        int i = 0; // Core index
        int active_cores = 0;  

        while (iter != end && i < 6) 
        {
            frame.cpu_load[i] = std::stoi((*iter)[1].str());
            ++active_cores;
            ++iter;
            ++i;
        }

        frame.active_cores = active_cores;
        // Handle inactive cores (e.g., "off")
        while (i < 6) 
        {
            frame.cpu_load[i] = -1; // Use -1 to indicate "inactive"
            ++i;
        }
    }

    // Extract GPU frequency
    if (std::regex_search(line, regexes.match, regexes.gpu_regex)) 
    {
        frame.gpu_freq = std::stoi(regexes.match[1].str());
    }

    // Extract CPU temperature
    if (std::regex_search(line, regexes.match, regexes.cpu_temp_regex)) 
    {
        frame.cpu_temp = std::stof(regexes.match[1].str());
    }

    // Extract GPU temperature
    if (std::regex_search(line, regexes.match, regexes.gpu_temp_regex)) 
    {
        frame.gpu_temp = std::stof(regexes.match[1].str());
    }

    // Extract VDD_IN
    if (std::regex_search(line, regexes.match, regexes.vdd_in_regex)) 
    {
        frame.vdd_in = std::stoi(regexes.match[1].str());
    }

    // Extract VDD_CPU_GPU_CV
    if (std::regex_search(line, regexes.match, regexes.vdd_cpu_gpu_cv_regex)) 
    {
        frame.vdd_cpu_gpu_cv = std::stoi(regexes.match[1].str());
    }

    // Extract VDD_SOC
    if (std::regex_search(line, regexes.match, regexes.vdd_soc_regex)) 
    {
        frame.vdd_soc = std::stoi(regexes.match[1].str());
    }

    // Reset the match object
    regexes.match = std::smatch();
}


void CaptureTegrastats() 
{
    // Set tegrastats interval
    FILE* pipe = popen(("tegrastats --interval " + std::to_string(TEGRASTATS_INTERVAL)).c_str(), "r");


    TegraTM frame; 
    RegexContainer regexes;

    if (!pipe) 
    {
        std::cerr << "Failed to open tegrastats pipe." << std::endl;
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), pipe)) // efficient blocking
    {
        ParseTegrastatsLine(line, regexes, frame);
        // PrintTegraTM(frame);
    }

    pclose(pipe);
}
