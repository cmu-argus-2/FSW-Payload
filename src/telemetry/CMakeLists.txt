


add_executable(TM_TEGRASTATS_PROCESS tegra_processor.cpp tegra.cpp)

# Link libraries to the executable
target_link_libraries(TM_TEGRASTATS_PROCESS PRIVATE
    spdlog::spdlog
    rt
)

