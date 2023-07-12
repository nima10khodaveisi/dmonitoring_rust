#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>
#include <cstring>

// #include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/opencv.hpp>

#include "NvInfer.h"

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


void DriverMonitoring::infer(const std::string& inputFilename) { 
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext()); 

    if (!context) { 
        return; 
    }
    cout << "Context has been created!\n";

    auto input_img_idx = mEngine->getBindingIndex("input_img"); 
    if (input_img_idx == -1) { 
        return; 
    }
    cout << "This is input image index " << input_img_idx <<  '\n';

    assert(mEngine->getBindingDataType(input_img_idx) == nvinfer1::DataType::kFLOAT);

    cv::Mat rgbImage = cv::imread(inputFilename, cv::IMREAD_COLOR); 
    if (rgbImage.empty()) {
        return; 
    }

    // try { 
    //     cv::resize(rgbImage, rgbImage.clone(), rgbImage.size(), cv::INTER_NEAREST_EXACT);
    // } catch(cv::Exception& e) { 
    //     const char* err_msg = e.what();
    //     std::cout << "exception caught: " << err_msg << std::endl;
    // }

    cout << "RGB image is fine " << rgbImage.size().width << ' ' << rgbImage.size().height << ' ' << rgbImage.channels() << endl; 

    cv::Mat yuvImage; 
    cv::cvtColor(rgbImage, yuvImage, cv::COLOR_BGR2YUV_I420); 
 
    cout << "YUV image data " << yuvImage.size().width << ' ' << yuvImage.size().height << ' ' << yuvImage.channels() << endl; 
 
    int bufferSize = MODEL_HEIGHT * MODEL_WIDTH;
    unsigned char* buffer = new unsigned char[bufferSize]; 

    std::memcpy(buffer, yuvImage.data, bufferSize); 

    cout << "Assigned buffer " << bufferSize << ' ' << yuvImage.total() << '\n';


}




int main() { 
    DriverMonitoring dm = DriverMonitoring("dmonitoring_model.engine"); 
    dm.infer("test.jpg");
}