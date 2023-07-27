#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <numeric>

// #include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "NvInfer.h"
#include <cuda_runtime.h>

using namespace std; 
using namespace nvinfer1; 


const int MODEL_WIDTH = 1440; 
const int MODEL_HEIGHT = 960; 


// I copied this from https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#c_topics
class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;



// Refrence: https://github.com/NVIDIA/TensorRT/blob/release/8.6/quickstart/SemanticSegmentation/tutorial-runtime.cpp

class DriverMonitoring { 
    public: 
        DriverMonitoring(const std::string& engineFilename);
        bool infer();

    private:
        std::string mEngineFilename;

        std::unique_ptr<nvinfer1::ICudaEngine> mEngine; 

}; 


DriverMonitoring::DriverMonitoring(const std::string& engineFilename): mEngine(nullptr) { 
    std::ifstream engineFile(engineFilename, std::ios::binary); 
    if (engineFile.fail()) { 
        return; 
    }        
    
    this->mEngineFilename = engineFilename; 

    engineFile.seekg(0, std::ifstream::end); 
    auto fsize = engineFile.tellg(); 
    engineFile.seekg(0, std::ifstream::beg); 

    std::vector<char> engineData(fsize); 
    engineFile.read(engineData.data(), fsize); 

    std::cout << "This is the size of engine file: " << fsize << " " << engineData.size() << endl; 

    // std::unique_ptr<nvinfer1::IRuntime> runtiسme{nvinfer1::createInferRuntime(logger)}; 
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    mEngine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr)); 
    assert(mEngine.get() != nullptr); 

    std::cout << "driver monitoring object has been initialized successfuly!\n";
}

bool DriverMonitoring::infer() { // input_buffer contains data for yuv frame 
    // netBuffer is the pointer to the input for the model
    auto input_img_index = mEngine->getBindingIndex("input_a"); 
    if (input_img_index == -1) {
        return false; 
    }

    auto calib_index = mEngine->getBindingIndex("input_b"); 
    if (calib_index == -1) { 
        return false; 
    }
    
    auto output_index = mEngine->getBindingIndex("output"); 
    if (output_index == -1) { 
        return false; 
    }


    cout << "input_img_index and calib_index and output_index are " << input_img_index << ' ' << calib_index << ' ' << output_index << endl; 
    int buffer_size = 10; 
    float a[] = {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
    float b[] = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};

    void* input_img_mem{nullptr};
    if (cudaMalloc(&input_img_mem, buffer_size * sizeof(float)) != cudaSuccess) { 
        cout << "Can not assign mem in cuda" << endl;
        return false; 
    }
    void* calib_mem{nullptr};
    if (cudaMalloc(&calib_mem, buffer_size * sizeof(float)) != cudaSuccess) { 
        cout << "Can not assign mem in cuda" << endl;
        return false; 
    }
    cout << "Successfuly assigned mem to input in cuda" << endl; 

    void* output_mem{nullptr}; 
    auto outputDims = mEngine->getBindingDimensions(output_index);
    auto outputSize = accumulate(outputDims.d, outputDims.d + outputDims.nbDims, 1, std::multiplies<int64_t>()); 
    
    if (cudaMalloc(&output_mem, 11 * sizeof(float)) != cudaSuccess) { 
        return false; 
    }
    cout << "Successfuly assigned mem to output in cuda" << endl; 

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        cout << "cuda stream creation failed." << std::endl;
        return false;
    }

    // return true;

    cout << "stream has been created!" << endl;

    // copy buffers to gpu
    if (cudaMemcpyAsync(input_img_mem, a, buffer_size * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        cout << "error in copying buffer from host to device" << endl; 
        return false; 
    }
    if (cudaMemcpyAsync(calib_mem, b, buffer_size * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        cout << "error in copying calib from host to device" << endl; 
        return false; 
    }
    cout << "copied data from buffers" << endl; 

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext()); 
    
    context->setTensorAddress("input_a", input_img_mem); 
    context->setTensorAddress("input_b", calib_mem); 
    context->setTensorAddress("output", output_mem); 

    bool status = context->enqueueV3(stream); 

    float* outputBuffer = new float[11];
    if (cudaMemcpyAsync(outputBuffer, output_mem, 11 * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        cout << "error in copying back output to host" << endl; 
        return false; 
    }
    cout << "success in copying output to host" << endl; 
    for (int i = 0; i < 11; ++i) { 
        cout << outputBuffer[i] << ' ';
    }
    cout << endl; 
    return true;
}




int main(int argc, char* argv[]) { 
    DriverMonitoring dm = DriverMonitoring("simple_sum_model.engine");
    dm.infer(); 

}
