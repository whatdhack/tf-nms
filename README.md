# tf-nms
Repository with several custom CUDA NMS ops for Tensorflow.

The code base liberally copies from Tensorflow  source and example code. In addition to CUDA, the code also uses CUB library.  The code base is primarily used for experimenting, hence has number of debug, printfs  etc are still active.  

# building
To build Tensorflow source is required, but a custom build is not. 
```
make clean; make
````

# running 

```
# tensorflow NMS  code
python ./customnms_test.py --algo v2 
# A simple custom NMS, that appears to be 2.4x better then the one in Tensorflow
python ./customnms_test.py --algo basic 
```
