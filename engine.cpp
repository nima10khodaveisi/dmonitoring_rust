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

#define REG_SCALE 0.25f


const int MODEL_WIDTH = 1440; 
const int MODEL_HEIGHT = 960; 

typedef struct DriverStateResult {
  float face_orientation[3];
  float face_orientation_std[3];
  float face_position[2];
  float face_position_std[2];
  float face_prob;
  float left_eye_prob;
  float right_eye_prob;
  float left_blink_prob;
  float right_blink_prob;
  float sunglasses_prob;
  float occluded_prob;
  float touching_wheel_prob;
  float paying_attention_prob; 
  float using_phone_prob;
  float distracted_prob; 
} DriverStateResult;


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
        DriverStateResult infer(float* buffer, const int buffer_size);

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

  switch (depth) {
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

  return "CV_" + r;
}

DriverStateResult DriverMonitoring::infer(float* buffer, const int buffer_size) { // input_buffer contains data for yuv frame 
    DriverStateResult model_res = {0}; 

    // netBuffer is the pointer to the input for the model
    auto input_img_index = mEngine->getBindingIndex("input_img"); 
    if (input_img_index == -1) {
        return model_res; 
    }

    auto calib_index = mEngine->getBindingIndex("calib"); 
    if (calib_index == -1) { 
        return model_res; 
    }
    
    auto output_index = mEngine->getBindingIndex("outputs"); 
    if (output_index == -1) { 
        return model_res; 
    }


    cout << "input_img_index and calib_index and output_index are " << input_img_index << ' ' << calib_index << ' ' << output_index << endl; 
    float calib[] = {0.0, 0.0, 0.0};

    void* input_img_mem{nullptr};
    if (cudaMalloc(&input_img_mem, buffer_size * sizeof(float)) != cudaSuccess) { 
        cout << "Can not assign mem in cuda" << endl;
        return model_res; 
    }
    void* calib_mem{nullptr};
    if (cudaMalloc(&calib_mem, 3 * sizeof(float)) != cudaSuccess) { 
        cout << "Can not assign mem in cuda" << endl;
        return model_res; 
    }
    cout << "Successfuly assigned mem to input in cuda" << endl; 

    void* output_mem{nullptr}; 
    auto outputDims = mEngine->getBindingDimensions(output_index);
    auto outputSize = accumulate(outputDims.d, outputDims.d + outputDims.nbDims, 1, std::multiplies<int64_t>()); 
    
    if (cudaMalloc(&output_mem, outputSize * sizeof(float)) != cudaSuccess) { 
        return model_res; 
    }
    cout << "Successfuly assigned mem to output in cuda" << endl; 

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        cout << "cuda stream creation failed." << std::endl;
        return model_res;
    }

    // return true;

    cout << "stream has been created!" << endl;

    // copy buffers to gpu
    if (cudaMemcpyAsync(input_img_mem, buffer, buffer_size * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        cout << "error in copying buffer from host to device" << endl; 
        return model_res; 
    }
    if (cudaMemcpyAsync(calib_mem, calib, 3 * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        cout << "error in copying calib from host to device" << endl; 
        return model_res; 
    }
    cout << "copied data from buffers" << endl; 

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext()); 
    
    context->setTensorAddress("input_img", input_img_mem); 
    context->setTensorAddress("calib", calib_mem); 
    context->setTensorAddress("outputs", output_mem); 

    bool status = context->enqueueV3(stream);

    cout << "done, status is " << status << ' ' << sizeof(output_mem) << ' ' << sizeof(buffer) << endl;
 

    float* output_buffer = new float[outputSize];
    if (cudaMemcpyAsync(output_buffer, output_mem, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        cout << "error in copying back output to host" << endl; 
        return model_res; 
    }
    cout << "success in copying output to host" << endl; 
    cout << (output_buffer[3] * 0.25f) << ' ' << (output_buffer[4] * 0.25f) << endl; 
    
    for (int person = 0; person < 1; ++person) { 
        int offset = person * 41;
        cout << "########## data for person=" << person + 1 << " ##########" << endl;  
        int cur = 0;  
        for (int d = 0; d < 3; ++d) { 
            model_res.face_orientation[d] = output_buffer[offset + cur++]; 
        }

        cout << "face " << output_buffer[offset + cur] * REG_SCALE << endl;
        model_res.face_position[0] = (output_buffer[offset + cur++] * REG_SCALE + 0.5) * MODEL_WIDTH;
        model_res.face_position[1] = (output_buffer[offset + cur++] * REG_SCALE + 0.5) * MODEL_HEIGHT;
        
        cout << "normalized face " << output_buffer[offset + cur++] << endl; 
        ++cur; 
        for (int d = 0; d < 3; ++d) { 
            model_res.face_orientation[d] = output_buffer[offset + cur++];
        }
        for (int d = 0; d < 2; ++d) { 
            model_res.face_orientation[d] = output_buffer[offset + cur++];
        }
        cout << "std normalized face " << output_buffer[offset + cur++] << endl; 
        
        model_res.face_prob = sigmoid(output_buffer[offset + cur++]);

        for (int eye = 0; eye < 2; ++eye) { 
            cout << "===== data for " << (eye == 0 ? "left": "right") << " eye =====" << endl; 
            for (int d = 0; d < 2; ++d) { 
                cout << "eye position " << output_buffer[offset + cur++] << endl; 
            }
            for (int d = 0; d < 2; ++d) { 
                cout << "eye size " << output_buffer[offset + cur++] << endl; 
            }
            for (int d = 0; d < 2; ++d) { 
                cout << "std eye position " << output_buffer[offset + cur++] << endl; 
            }
            for (int d = 0; d < 2; ++d) { 
                cout << "std eye size " << output_buffer[offset + cur++] << endl; 
            }
            (eye == 0? model_res.left_eye_prob: model_res.right_eye_prob) = sigmoid(output_buffer[offset + cur++]);
            (eye == 0? model_res.left_blink_prob: model_res.right_blink_prob) = sigmoid(output_buffer[offset + cur++]);
        }
        cout << "======================" << endl; 
        model_res.sunglasses_prob = sigmoid(output_buffer[offset + cur++]);
        model_res.occluded_prob = sigmoid(output_buffer[offset + cur++]);
        model_res.touching_wheel_prob = sigmoid(output_buffer[offset + cur++]);
        model_res.paying_attention_prob = sigmoid(output_buffer[offset + cur++]);  
        cur += 2; // ignore deprecated 
        model_res.using_phone_prob =  sigmoid(output_buffer[offset + cur++]);
        model_res.distracted_prob = sigmoid(output_buffer[offset + cur++]);
        cout << "###################################################" << endl; 
    }
    return model_res;
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

    cout << "width and height of video are " << frame_width << ' ' << frame_height << endl; 

    cv::VideoWriter video("result.mkv", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(1440, 960));

    DriverMonitoring dm = DriverMonitoring("dmonitoring_model.engine"); 

    while (true) { 
        cv::Mat rgb_frame; 
        cap >> rgb_frame;

        cout << "this is frame size " << rgb_frame.size() << ' ' << type2str(rgb_frame.type()) << endl; 
        
        if (rgb_frame.empty()) { 
            break; 
        }

        cv::Mat resized_frame;
        cv::resize(rgb_frame, resized_frame, cv::Size(1440, 960));
        cout << "this is resized frame size " << resized_frame.size() << endl; 

        cv::Mat yuv_frame; 
        cv::cvtColor(resized_frame, yuv_frame, cv::COLOR_BGR2YUV); 

        cout << "this is yuv frame size " << yuv_frame.size() << ' ' << type2str(yuv_frame.type()) << endl; 

        std::vector<cv::Mat> channels;
        cv::split(yuv_frame, channels); 
        cv::Mat y = channels[0];
        cout << "this is y shape " << y.size() << endl; 

        y.convertTo(y, CV_32F); //convert to float32
        y /= 255.;


        const int buffer_size = int(MODEL_HEIGHT * MODEL_WIDTH);
        float* buffer = new float[buffer_size]; 

        std::memcpy(buffer, y.data, buffer_size); 

        DriverStateResult res = dm.infer(buffer, buffer_size);

        vector<string> frame_data = {
            "using_phone_prob: " + to_string(res.using_phone_prob),
            "sunglasses_prob: " + to_string(res.sunglasses_prob),
            "distracted_prob: " + to_string(res.distracted_prob)
        };

        for (size_t i = 0; i < frame_data.size(); ++i) {
            cv::Size textSize = cv::getTextSize(frame_data[i], cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, 0);

            int x = 50;
            int y = (i + 1) * (textSize.height + 50);

            cv::putText(resized_frame, frame_data[i], cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }
        cv::circle(resized_frame, cv::Point(res.face_position[0], res.face_position[1]), 70, cv::Scalar(0, 0, 255), 10);

        video.write(resized_frame); 

        // break; 
    }

    video.release();
}
