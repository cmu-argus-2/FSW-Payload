

add_library(core
    task.cpp
    data_handling.cpp
    thread_pool.cpp
    utils.cpp
    timing.cpp
    errors.cpp
)

# target_link_libraries(core <dependencies>)
target_link_libraries(core
    PRIVATE spdlog::spdlog nlohmann_json::nlohmann_json
)