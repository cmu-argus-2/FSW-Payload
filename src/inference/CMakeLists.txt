if(CUDA_ENABLED)
    add_library(inference
        runtimes.cpp
        orchestrator.cpp
    )
    target_link_libraries(inference
        PRIVATE 
            core 
            ${OpenCV_LIBS}
            ${NVINFER_LIBRARY}
            ${NVINFER_PLUGIN_LIBRARY}
            ${CUDART_LIBRARY}
    )
else()
    add_library(inference INTERFACE)
endif()