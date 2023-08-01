# dmonitoring_rust

## How to run the project?
To get the results, you just have to run engine.cpp file, other files in the repo is just for development purposes. Note that if don't have access to Nvidia gpu, you can ask me to setup the server for you (I rent an RTX 3080 from [vast.ai](https://vast.ai)). First you need pull the pytorch docker from nvidia: 

```
docker pull nvcr.io/nvidia/pytorch:23.06-py3
```

Then run the docker using following command: 

```
docker run --rm --gpus all -ti --volume $HOME/trt_release/workspace/trt_release --net host nvcr.io/nvidia/pytorch:23.06-py3
```

Then you have to install opencv on your docker: 

```
apt install libopencv-dev
```

Finally, you can run the model using following commands:

```
make clean
make
./engine --input=interior_center_day.mvk
```

## What are the results?

First I created the engine file from onnx file using trtexec command. The DriverMonitoring class, takes an engine file as input and create a TensorRT engine. Then, the infer function takes a buffer which is pointed to the Y channel of our frame (which is an image represented in YUV 4:2:0) and handles copying data from cpu to gpu and running the model and finally returns the result. Unfortunately, the results are not correct. 

## Why the results are not correct? 

The short answer is I do not know. But in this section I want to explain what I did to fix this. When I finished the implementations I found that the results are not correct. There are a few steps before getting results which I had to test and make sure that part is right. I will list them all below: 

* Copy data from cpu to gpu and running the model: As you can see, there is a file simple_sum_model.onnx which takes two Tensors and returns sum of them. I created the engine file from this simple model and tested the resutls and it was okay, so I believe this part is fine. Also, this ensures that my engine file has been created correctly. 
* Preparing the input and parsing the output for DM model: So this part has a problem. I almost read all codes of openpilot related to DM model, how they prepare the input, how they parse it and how they use it. I even emailed one of their authors about the input of model and they updated the readme ([here](https://github.com/commaai/openpilot/pull/29188)). I created a simple python file called onnx_runner which takes an onnx file and a video and prepare each frame for the input and print the results. So, we know that the input for the model is an image with width=1440 and height=960, then we should convert this image to yuv and just take the y channel and give it to the model. Then we have parsing data. I tried to parse data for face position, they first multiply the output by some REG_SCALE ([here](https://github.com/commaai/openpilot/blob/1e1cc638d8e9fa408a02a57d689c13c2219d823d/selfdrive/modeld/models/dmonitoring.cc#L34)) and then calculate the x and y of the driver image like [this](https://github.com/commaai/openpilot/blob/1e1cc638d8e9fa408a02a57d689c13c2219d823d/selfdrive/monitoring/driver_monitor.py#L84). I did the exact parsing but I am not getting the right results. 

## What is next? 
I put a lot of efforts and time to make the model work, for some reason I couldn't. I may be able to contact one person in commaai and get some help from them, with this simple onnx_runner.py script everyone can see what is wrong very quickly if they're familiar with the model. 
