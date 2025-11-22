

add_library(inference
    runtimes.cpp
    orchestrator.cpp
)

target_link_libraries(inference
    PRIVATE 
        core 
        ${OpenCV_LIBS}
        ${NVINFER_LIBRARY}
        ${CUDART_LIBRARY}
)