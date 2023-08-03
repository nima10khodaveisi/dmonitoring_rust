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

First I created the engine file from onnx file using trtexec command. The DriverMonitoring class, takes an engine file as input and create a TensorRT engine. Then, the infer function takes a buffer which is pointed to the Y channel of our frame (which is an image represented in YUV 4:2:0) and handles copying data from cpu to gpu and running the model and finally returns the result.
