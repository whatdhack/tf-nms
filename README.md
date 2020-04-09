# tf-nms
Repository with several custom CUDA NMS ops for Tensorflow.

The code base liberally copies from Tensorflow  source and example code. In addition to CUDA, the code also uses CUB library.  The code base is primarily used for experimenting, hence has number of debug, printfs  etc still in place.  

# building
To build, Tensorflow source is required, but a custom Tensorflow build is not. 

Requires: 
  1. Nvidia cub headers ( make require customization of Makefile).

```
make clean; make
````

# running 
Currently 2 algorithms are available (others are there, but not added to the CLI).  The algorithm called "v2" is the one found in Tensorflow 1.15.  The algorithm called "basic" is an alternative algorithm that appears to have better runtime numbers on GPUs.

```
# tensorflow NMS  code
python ./customnms_test.py --algo v2 
# A simple custom NMS, that appears to be 2.4x better then the one in Tensorflow
python ./customnms_test.py --algo basic 
```
# performance
The "basic" algorithm is about 2.4x faster than the "v2" algorithm found in Tensorflow 1.15.

# references
1. Reflections on Non Maximum Suppression (NMS), https://medium.com/@whatdhack/reflections-on-non-maximum-suppression-nms-d2fce148ef0a
