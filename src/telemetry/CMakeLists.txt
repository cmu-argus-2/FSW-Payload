
add_library(telemetry
    telemetry.cpp
    tegra.cpp
)

target_link_libraries(telemetry PRIVATE
    rt
)


##### Tegrastats Processor 
set(TM_TEGRASTATS "TM_TEGRASTATS")
add_compile_definitions(TM_TEGRASTATS=\"${TM_TEGRASTATS}\")

add_executable(TM_TEGRASTATS tegra_processor.cpp tegra.cpp)

# Link libraries to the executable
target_link_libraries(TM_TEGRASTATS PRIVATE
    ${PAYLOAD_SPDLOG_TARGET}
    rt
)