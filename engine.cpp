#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cassert>

#include "NvInfer.h"

using namespace std; 
using namespace nvinfer1; 


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
        DriverMonitoring(const std::string& engineFileName); 

    private:
        std::string mEngineFilename;

        std::unique_ptr<nvinfer1::ICudaEngine> mEngine; 

}; 


DriverMonitoring::DriverMonitoring(const std::string& engineFileName): mEngine(nullptr) { 
    std::ifstream engineFile(engineFileName, std::ios::binary); 
    if (engineFile.fail()) { 
        return; 
    }        
    
    this->mEngineFilename = engineFileName; 

    engineFile.seekg(0, std::ifstream::end); 
    auto fsize = engineFile.tellg(); 
    engineFile.seekg(0, std::ifstream::beg); 

    std::vector<char> engineData(fsize); 
    engineFile.read(engineData.data(), fsize); 

    std::cout << "This is the size of engine file: " << fsize << " " << engineData.size() << endl; 

    // std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)}; 
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    // mEngine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr)); 

    // assert(mEngine.get() != nullptr); 


}




int main() { 
    DriverMonitoring dm = DriverMonitoring("dmonitoring_model.engine"); 
}