


add_executable(tegrastats_process tegra_processor.cpp tegra.cpp)

# Link libraries to the executable
target_link_libraries(tegrastats_process PRIVATE
    spdlog::spdlog
    rt
)

# Add compiler warnings (as errors) and other warning flags
target_compile_options(tegrastats_process PRIVATE
    -Werror
    -Wall
    -Wextra
    -Wconversion
    -Wsign-conversion
)

