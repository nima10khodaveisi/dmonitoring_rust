# dmonitoring_rust

First you need pull the pytorch docker from nvidia: 

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
./engine --input=test.jpg
```
