#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#define NUM_CLASSES 16


// https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html#runtime

using namespace nvinfer1;


// Logger for TensorRT
class Logger : public ILogger 
{
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) std::cout << "[TRT] " << msg << std::endl;
    }
};

// Load the serialized TensorRT engine from the .trt file
std::vector<char> LoadEngine(const std::string& engine_path) 
{
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error: Failed to open engine file " << engine_path << std::endl;
        exit(1);
    }
    file.seekg(0, std::ios::end); // Move to the end of the file
    size_t size = file.tellg(); // Get the size of the file
    file.seekg(0, std::ios::beg); // Move back to the beginning of the file
    std::vector<char> engine_data(size); // Create a vector to hold the engine data
    file.read(engine_data.data(), size);
    return engine_data;
}

// Helper function to calculate the memory size of a tensor
size_t GetMemorySize(const nvinfer1::Dims& dims, size_t element_size) 
{
    size_t size = element_size;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}


// Normalize image like torchvision.transforms.Normalize
void NormalizeImage(cv::Mat& img, const std::vector<float>& mean, const std::vector<float>& std) 
{
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);  // Convert to float, scale to [0,1]
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std[i];  // Normalize
    }
    cv::merge(channels, img);
}

cv::Mat Preprocess(cv::Mat& image) 
{
    // COnvert BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    
    // 1. Resize
    cv::resize(image, image, cv::Size(224, 224), 0, 0, cv::INTER_AREA); 
    // Note: cv::INTER_AREA is the best one I found closest to torchvision.transforms.Resize (which uses PIL resize's default)
    // Ideally, we would want to avoid that because it has *slight* pixel differences due to their implementation

    // Print shape 
    // std::cout << "Image shape: " << image.rows << "x" << image.cols << "x" << image.channels() << std::endl; 

    // 2. ToTensor (Convert to float and scale to [0, 1]) 
    // Convert from uint8 [0, 255] to float32 [0, 1]
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F, 1.0 / 255.0);  // Correctly normalize
    // Convert HWC (224x224x3) to CHW (3x224x224)
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);  // Split into individual float32 channels
    
    // restacking now after the normalization to avoid re-splitting
    // cv::Mat chw_img;
    // cv::vconcat(channels, chw_img);  // Stack channels vertically (CHW format)

    // 3. Normalize per channel
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    // Apply normalization per channel
    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean[c]) / std[c];  // Element-wise operation
    }


    // stack them back
    cv::Mat chw_img;
    cv::vconcat(channels, chw_img);  // Stack channels vertically (CHW format)


    /*// print first 10 elements of each channel
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "First 10 elements of each channel:\n";

    for (int c = 0; c < 3; c++)  // Loop over 3 channels
    {
        std::cout << "Channel " << c << ": ";
        for (int j = 0; j < 10; j++) 
        {   
            std::cout << chw_img.at<float>(c * 224 + 0, j) << " "; // Correct indexing
        }
        std::cout << std::endl;
    }

    // Print the first 5x5 region of Channel 0
    std::cout << "Channel 0:\n";
    for (int i = 0; i < 5; i++) 
    {
        for (int j = 0; j < 5; j++) 
        {
            std::cout << chw_img.at<float>(0 * 224 + i, j) << " "; // Correct indexing
        }
        std::cout << std::endl;
    }*/

    return chw_img.clone();
}

int main(int argc, char** argv) 
{
    
    if (argc != 3) 
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    std::string image_path = argv[2];
    
    
    // Load image using OpenCV
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) 
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    /* PREPROCESSING */
    cv::Mat chw_img = Preprocess(image);
    std::cout << "Image preprocessed to CHW format with shape: " 
              << chw_img.rows << "x" << chw_img.cols << "x" << chw_img.channels() << std::endl;
    
    // Load the TensorRT engine
    std::vector<char> engine_data = LoadEngine(model_path);
    Logger logger;

    // Create runtime and deserialize engine
    std::unique_ptr<IRuntime> mRuntime{nvinfer1::createInferRuntime(logger)};
    if (!mRuntime) {
        std::cerr << "Error: Failed to create runtime" << std::endl;
        return 1;
    }

    std::unique_ptr<ICudaEngine> mEngine(mRuntime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!mEngine) {
        std::cerr << "Error: Failed to create CUDA engine" << std::endl;
        return 1;
    }

    // Create execution context
    std::unique_ptr<IExecutionContext> context{mEngine->createExecutionContext()};
    if (!context) 
    {
        std::cerr << "Error: Failed to create execution context" << std::endl;
        return 1;
    }

    // Verify tensor names
    std::cout << "Available tensors in engine:" << std::endl;
    for (int i = 0; i < mEngine->getNbIOTensors(); i++) 
    {
    const char* tensor_name = mEngine->getIOTensorName(i);
    nvinfer1::TensorIOMode mode = mEngine->getTensorIOMode(tensor_name);
    
    std::cout << "Tensor " << i << ": " << tensor_name
              << " (" << (mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output") << ")"
              << std::endl;
    }

    char const* input_name = "l_x_";
    char const* output_name = "sigmoid_1";
    assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);
    assert(mEngine->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);


    // Set input 
    int height = 224;
    int width = 224;
    int batch_size = 1;
    int nb_channel = 3;
    auto input_dims = nvinfer1::Dims4{batch_size, nb_channel, height, width};
    context->setInputShape(input_name, input_dims);
    auto output_dims = context->getTensorShape(output_name);
    

    // Allocate GPU memory for input and output
    size_t input_size = 1 * nb_channel * height * width * sizeof(float);
    size_t output_size = NUM_CLASSES * sizeof(float);  // 16-class output

    void* gpu_buffers[2];
    cudaMalloc(&gpu_buffers[0], input_size);
    cudaMalloc(&gpu_buffers[1], output_size);

    // Bind input and output memory
    context->setTensorAddress(input_name, gpu_buffers[0]);
    context->setTensorAddress(output_name, gpu_buffers[1]);

    // Copy input data to GPU
    cudaMemcpy(gpu_buffers[0], chw_img.ptr<float>(), input_size, cudaMemcpyHostToDevice);
    std::cout << "Image successfully preprocessed and copied to CUDA memory!" << std::endl;

    // Create CUDA stream
    cudaStream_t stream; // CUDA stream for asynchronous execution
    cudaStreamCreate(&stream); 

    // Run asynchronous inference using `enqueueV3()`
    bool status = context->enqueueV3(stream);
    if (!status)
    {
        std::cerr << "Error: Failed to execute inference." << std::endl;
        return 1;
    }

    auto output_buffer = std::unique_ptr<float[]>{new float[NUM_CLASSES]};  // Allocate buffer for output
    cudaMemcpyAsync(output_buffer.get(), gpu_buffers[1], output_size, cudaMemcpyDeviceToHost, stream); // Copy output back to CPU

    cudaStreamSynchronize(stream);  // Wait for inference to finish

    

    // Print whole output
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Output: " << std::endl;
    for (int i = 0; i < NUM_CLASSES; i++) 
    {
        std::cout << output_buffer[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(gpu_buffers[0]);
    cudaFree(gpu_buffers[1]);
    context.reset();
    mEngine.reset();
    mRuntime.reset();


    return 0;
}

// g++ -o run_trt run_trt.cpp -lnvinfer -lcudart -std=c++17 -I/usr/include/opencv4/opencv -I/usr/include/opencv4
