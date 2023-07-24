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
        void infer(const std::string& inputFile);

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

float sigmoid(float x) { 
    return 1 / (1 + exp(x));
}

void DriverMonitoring::infer(const std::string& inputFilename) { 
    // auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext()); 

    cv::Mat rgbImage = cv::imread(inputFilename, cv::IMREAD_COLOR); 
    if (rgbImage.empty()) {
        return; 
    }

    cout << "RGB image is fine " << rgbImage.size().width << ' ' << rgbImage.size().height << ' ' << rgbImage.channels() << endl; 

    cv::Mat yuvImage; 
    cv::cvtColor(rgbImage, yuvImage, cv::COLOR_BGR2YUV_I420); 

    // Yuv image data 
    int width = yuvImage.size().width; 
    int height = yuvImage.size().height;
    int stride = width; // This is because our picutre is completely inside the jpeg file

    cout << "YUV image data " << width << ' ' << height << ' ' << yuvImage.channels() << endl; 
    cout << "YUV desired size " << MODEL_WIDTH << ' ' << MODEL_HEIGHT << ' ' << yuvImage.channels() << endl; 

    int streamBufferSize = width * height;
    uint8_t* streamBuffer = new uint8_t[streamBufferSize]; 

    std::memcpy(streamBuffer, yuvImage.data, streamBufferSize); 

    FILE *sdump_yuv_file = fopen("rawdump_stream.yuv", "wb");
    fwrite(streamBuffer, streamBufferSize, sizeof(uint8_t), sdump_yuv_file);
    fclose(sdump_yuv_file);

    cout << "Assigned buffer " << streamBufferSize << ' ' << yuvImage.total() << '\n';

    int netBufferSize = MODEL_HEIGHT * MODEL_WIDTH; 
    uint8_t* netBuffer = new uint8_t[netBufferSize];

    int v_off = height - MODEL_HEIGHT;
    int h_off = (width - MODEL_WIDTH) / 2;
    cout << "h_off and v_off are " << h_off << ' ' << v_off << endl; 

    for (int r = 0; r < MODEL_HEIGHT; ++r) { 
        memcpy(netBuffer + r * MODEL_WIDTH, streamBuffer + stride * v_off + r * stride + h_off, MODEL_WIDTH);
    }

    cout << "Resizing completed!" << endl; 
    FILE *dump_yuv_file = fopen("rawdump.yuv", "wb");
    fwrite(netBuffer, netBufferSize, sizeof(uint8_t), dump_yuv_file);
    fclose(dump_yuv_file);

    // netBuffer is the pointer to the input for the model
    auto input_img_index = mEngine->getBindingIndex("input_img"); 
    if (input_img_index == -1) {
        return; 
    }

    auto calib_index = mEngine->getBindingIndex("calib"); 
    if (calib_index == -1) { 
        return; 
    }
    
    auto output_index = mEngine->getBindingIndex("outputs"); 
    if (output_index == -1) { 
        return; 
    }


    cout << "input_img_index and calib_index and output_index are " << input_img_index << ' ' << calib_index << ' ' << output_index << endl; 
    float calib[] = {1.0, 2, 3};

    void* input_img_mem{nullptr};
    if (cudaMalloc(&input_img_mem, netBufferSize * sizeof(float)) != cudaSuccess) { 
        cout << "Can not assign mem in cuda" << endl;
        return; 
    }
    void* calib_mem{nullptr};
    if (cudaMalloc(&calib_mem, sizeof(calib) * sizeof(float)) != cudaSuccess) { 
        cout << "Can not assign mem in cuda" << endl;
        return; 
    }
    cout << "Successfuly assigned mem to input in cuda" << endl; 

    void* output_mem{nullptr}; 
    auto outputDims = mEngine->getBindingDimensions(output_index);
    auto outputSize = accumulate(outputDims.d, outputDims.d + outputDims.nbDims, 1, std::multiplies<int64_t>()) * sizeof(float); 
    
    if (cudaMalloc(&output_mem, outputSize) != cudaSuccess) { 
        return; 
    }
    cout << "Successfuly assigned mem to output in cuda" << endl; 

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        cout << "cuda stream creation failed." << std::endl;
        return;
    }

    cout << "stream has been created!" << endl;  

    // copy buffers to gpu
    if (cudaMemcpyAsync(input_img_mem, (float*) netBuffer, netBufferSize / sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        cout << "error in copying data from host to device" << endl; 
        return; 
    }
    if (cudaMemcpyAsync(calib_mem, calib, sizeof(calib), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        cout << "error in copying data from host to device" << endl; 
        return; 
    }
    cout << "copied data from buffers" << endl; 

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext()); 
    
    context->setTensorAddress("input_img", input_img_mem); 
    context->setTensorAddress("calib", calib_mem); 
    context->setTensorAddress("outputs", output_mem); 


    bool status = context->enqueueV3(stream);

    cout << "done, status is " << status << endl; 

    float* outputBuffer = new float[outputSize];
      if (cudaMemcpyAsync(outputBuffer, output_mem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        cout << "error in copying back output to host" << endl; 
        return; 
    }
    cout << "success in copying output to host" << endl; 
    for (int i = 0; i < 84; ++i) { 
        cout << sigmoid(outputBuffer[i]) << ' '; 
    }
    cout << endl; 
}




int main() { 
    DriverMonitoring dm = DriverMonitoring("dmonitoring_model.engine"); 
    dm.infer("test.jpg");
}
