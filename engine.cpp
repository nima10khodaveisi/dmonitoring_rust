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
        bool infer(const std::string& inputFile);

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

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

bool DriverMonitoring::infer(const std::string& inputFilename) { 

    cv::Mat rgbImage = cv::imread(inputFilename, cv::IMREAD_COLOR); 
    if (rgbImage.empty()) {
        return false; 
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

    // int v_off = height - MODEL_HEIGHT;
    // int h_off = (width - MODEL_WIDTH) / 2;
    // cout << "h_off and v_off are " << h_off << ' ' << v_off << endl; 

    // for (int r = 0; r < MODEL_HEIGHT; ++r) { 
    //     memcpy(netBuffer + r * MODEL_WIDTH, streamBuffer + stride * v_off + r * stride + h_off, MODEL_WIDTH);
    // }

    // This is the size of image such that dw * dh * 1.5 ~ 960 * 1440 
    int dw = 1172; 
    int dh = 884; 

    int sx = (width - dw) / 2; 
    int sy = (height - dh) / 2;

    uint8_t *yStreamBuffer = streamBuffer + sy * width + sx; 

    // copy y 
    for (int r = 0; r < dh; ++r) { 
        memcpy(netBuffer + r * dw, yStreamBuffer + width * r, dw); 
    }

    // copy u and v 
    uint8_t *uNetBuffer = netBuffer + dw * dh; 
    uint8_t *uStreamBuffer = streamBuffer + width * height + (sy / 2) * width ; 
    uint8_t *vStreamBuffer = uStreamBuffer + (width / 2) * (height / 2); 
    for (int r = 0; r < dh / 2; ++r) { 
        // u 
        memcpy(uNetBuffer + r * dw, uStreamBuffer + width * r + sx / 2, dw / 2); 
        memcpy(uNetBuffer + (r + 1) * dw, vStreamBuffer + width * r + sx / 2, dw / 2); 
    }

    cout << "Resizing completed!" << endl; 
    FILE *dump_yuv_file = fopen("rawdump.yuv", "wb");
    fwrite(netBuffer, netBufferSize, sizeof(uint8_t), dump_yuv_file);
    fclose(dump_yuv_file);

    // netBuffer is the pointer to the input for the model
    auto input_img_index = mEngine->getBindingIndex("input_img"); 
    if (input_img_index == -1) {
        return false; 
    }

    auto calib_index = mEngine->getBindingIndex("calib"); 
    if (calib_index == -1) { 
        return false; 
    }
    
    auto output_index = mEngine->getBindingIndex("outputs"); 
    if (output_index == -1) { 
        return false; 
    }


    cout << "input_img_index and calib_index and output_index are " << input_img_index << ' ' << calib_index << ' ' << output_index << endl; 
    float calib[] = {1.0, 2, 3};

    void* input_img_mem{nullptr};
    if (cudaMalloc(&input_img_mem, netBufferSize * sizeof(float)) != cudaSuccess) { 
        cout << "Can not assign mem in cuda" << endl;
        return false; 
    }
    void* calib_mem{nullptr};
    if (cudaMalloc(&calib_mem, sizeof(calib) * sizeof(float)) != cudaSuccess) { 
        cout << "Can not assign mem in cuda" << endl;
        return false; 
    }
    cout << "Successfuly assigned mem to input in cuda" << endl; 

    void* output_mem{nullptr}; 
    auto outputDims = mEngine->getBindingDimensions(output_index);
    auto outputSize = accumulate(outputDims.d, outputDims.d + outputDims.nbDims, 1, std::multiplies<int64_t>()) * sizeof(float); 
    
    if (cudaMalloc(&output_mem, outputSize) != cudaSuccess) { 
        return false; 
    }
    cout << "Successfuly assigned mem to output in cuda" << endl; 

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        cout << "cuda stream creation failed." << std::endl;
        return false;
    }

    cout << "stream has been created!" << endl;  

    // copy buffers to gpu
    if (cudaMemcpyAsync(input_img_mem, (float*) netBuffer, netBufferSize / sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        cout << "error in copying data from host to device" << endl; 
        return false; 
    }
    if (cudaMemcpyAsync(calib_mem, calib, sizeof(calib), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        cout << "error in copying data from host to device" << endl; 
        return false; 
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
        return false; 
    }
    cout << "success in copying output to host" << endl; 
    return true; 
    for (int person = 0; person < 2; ++person) { 
        int offset = person * 41;
        cout << "########## data for person=" << person + 1 << " ##########" << endl;  
        int cur = 0;  
        for (int d = 0; d < 3; ++d) { 
            cout << "face orientition " << outputBuffer[offset + cur++] << endl; 
        }
        for (int d = 0; d < 2; ++d) { 
            cout << "face position " << outputBuffer[offset + cur++] << endl; 
        }
        cout << "normalized face " << outputBuffer[offset + cur++] << endl; 
        for (int d = 0; d < 3; ++d) { 
            cout << "std face orientition " << outputBuffer[offset + cur++] << endl; 
        }
        for (int d = 0; d < 2; ++d) { 
            cout << "std face position " << outputBuffer[offset + cur++] << endl; 
        }
        cout << "std normalized face " << outputBuffer[offset + cur++] << endl; 
        
        cout << "face visible probability " << sigmoid(outputBuffer[offset + cur++]) << endl;

        for (int eye = 0; eye < 2; ++eye) { 
            cout << "===== data for " << (eye == 0 ? "left": "right") << " eye =====" << endl; 
            for (int d = 0; d < 2; ++d) { 
                cout << "eye position " << outputBuffer[offset + cur++] << endl; 
            }
            for (int d = 0; d < 2; ++d) { 
                cout << "eye size " << outputBuffer[offset + cur++] << endl; 
            }
            for (int d = 0; d < 2; ++d) { 
                cout << "std eye position " << outputBuffer[offset + cur++] << endl; 
            }
            for (int d = 0; d < 2; ++d) { 
                cout << "std eye size " << outputBuffer[offset + cur++] << endl; 
            }
            cout << "eye visible probability " << sigmoid(outputBuffer[offset + cur++]) << endl; 
            cout << "eye closed probability " << sigmoid(outputBuffer[offset + cur++]) << endl; 
        }
        cout << "======================" << endl; 
        cout << "wearing sunglass probabilty " << sigmoid(outputBuffer[offset + cur++]) << endl; 
        cout << "face occlueded probability " << sigmoid(outputBuffer[offset + cur++]) << endl; 
        cout << "touching wheel probability " << sigmoid(outputBuffer[offset + cur++]) << endl; 
        cout << "paying attention probability " << sigmoid(outputBuffer[offset + cur++]) << endl; 
        cur += 2; // ignore deprecated 
        cout << "using phone probability " << sigmoid(outputBuffer[offset + cur++]) << endl;
        cout << "distracted probablity " << sigmoid(outputBuffer[offset + cur++]) << endl; 
        cout << "###################################################" << endl; 
    }
    return true; 
}




int main(int argc, char* argv[]) { 
    string inputFileName = "interior_center_day.mkv";
    if (argc < 2) {
        cout << "running using default args, --input=test.jpg, to change this you can run the code with ./engine --input=foo.bar" << endl; 
    } else { 
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.substr(0, 8) == "--input=") {
                inputFileName = arg.substr(8);
                break;
            }
        }
    }

    cv::VideoCapture cap(inputFileName);
    if (!cap.isOpened()) { 
        cout << "error, file not found!" << endl; 
        return 0;  
    }
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int pixel_format = static_cast<int>(cap.get(cv::CAP_PROP_FORMAT));

    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    if (pixel_format == CV_8UC3) {
        cout << "video has 3 channels" << endl;
    } else if (pixel_format == CV_8UC1) {
        cout << "video has 1 channel" << endl;
    } else {
        cout << "video format is not valid" << endl;
        return 0;
    }

    cout << "width and height of video are " << frame_width << ' ' << frame_height << endl; 

    int cnt = 0; 

    while (true) { 
        cv::Mat rgb_frame, y_plane, u_plane, v_plane; 
        cap >> rgb_frame; 

        cout << "this is frame size " << rgb_frame.size() << ' ' << type2str(rgb_frame.type()) << endl; 

        // resize rgb image to 1152 * 800 then width * height * 1.5 = 1440 * 960 
        cout << "shit" << endl; 
        cv::Mat rgb_resized = cv::Mat(800, 1152, 16);  
        cout << "done " << endl; 
        cout << rgb_resized.size() << ' ' << type2str(rgb_resized.type()) << endl; 
        cout << "wow " << endl; 
        auto sz = cv::Size((int)1152, (int)800); 
        cout << "oh oh oh" << endl; 
        cv::resize(rgb_frame, rgb_resized, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        cout << "eeee" << endl;  
        cout << "done " << rgb_resized.size() << ' ' << type2str(rgb_resized.type()) << endl; 

        /* 
            frame is bgr, we have to convert it to yuv.
            we can try to read the video from yuv format directly. 
            https://stackoverflow.com/questions/27496698/opencv-capture-yuyv-from-camera-without-rgb-conversion
        */
        
        if (rgb_frame.empty()) { 
            break; 
        }
        ++cnt; 

        cv::Mat yuv_frame; 
        cv::cvtColor(rgb_frame, yuv_frame, cv::COLOR_BGR2YUV_I420); 

        cout << "this is yuv frame size " << yuv_frame.size() << type2str(yuv_frame.type()) << endl;  

        uint8_t* buffer = new uint8_t[640 * 720]; 

        std::memcpy(buffer, yuv_frame.data, 640 * 720); 

        FILE *sdump_yuv_file = fopen("frame.yuv", "wb");
        fwrite(buffer, 640 * 720, sizeof(uint8_t), sdump_yuv_file);
        fclose(sdump_yuv_file);
  

        // cv::imwrite("frame.jpg", frame);

        // std::vector<cv::Mat> planes;
        // cv::split(frame, planes);
        // y_plane = planes[0];
        // u_plane = planes[1];
        // v_plane = planes[2];

        // uint8_t* streamBuffer = new uint8_t[640 * 480]; 

        // std::memcpy(streamBuffer, y_plane.data, 640 * 480); 

        // FILE *sdump_yuv_file = fopen("frame.yuv", "wb");
        // fwrite(streamBuffer, 640 * 480, sizeof(uint8_t), sdump_yuv_file);
        // fclose(sdump_yuv_file);

        // cout << u_plane.size() << ' ' << v_plane.size() << endl; 

        break; 
    }

    // cout << "this is number of frames: " << cnt << ' ' << total_frames << endl; 


    // DriverMonitoring dm = DriverMonitoring("dmonitoring_model.engine"); 
    // bool status = dm.infer(inputFileName);
    // if (!status) { 
    //     cout << "some error happened" << endl; // todo: show the error exactly!
    // }
}
