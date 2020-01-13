CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python
NVARCH = sm_60 
#CUSTOMNMS_SRCS = $(wildcard *.h) $(wildcard *.cc)
CUSTOMNMS_SRCS =  $(wildcard *.cc)
CUDA_SRCS = $(wildcard *.cu)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

#CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11 -g
#NVCCFLAGS = ${TF_CFLAGS} -I /home/sgoswami/experiments/tensorflow -g
CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
NVCCFLAGS = ${TF_CFLAGS} -I /home/sgoswami/experiments/tensorflow
LDFLAGS = -shared ${TF_LFLAGS}

CUSTOMNMS_GPU_OBJS = $(patsubst %.cu, %.cu.o, $(CUDA_SRCS)) 
CUSTOMNMS_CPU_OBJS = $(patsubst %.cc, %.o, $(CUSTOMNMS_SRCS)) 
CUSTOMNMS_ALL_GPU_OBJ = gpu_code.o 
CUSTOMNMS_ALL_GPU_LIB = gpu_code.a
CUSTOMNMS_TARGET_LIB = custom_nms_ops.so


all: customnms_op 

customnms_op: $(CUSTOMNMS_TARGET_LIB)

%.cu.o: %.cu
	echo $(NVCCFLAGS)
	echo $(TF_LFLAGS)
	#	$(NVCC) -arch=$(NVARCH) -std=c++11 -dc -o $@ $^  $(NVCCFLAGS) $(TF_LFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr
	$(NVCC) -arch=$(NVARCH) -std=c++11 -c -o $@ $^  $(NVCCFLAGS) $(TF_LFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

$(CUSTOMNMS_ALL_GPU_OBJ) : $(CUSTOMNMS_GPU_OBJS)
	$(NVCC) -arch=$(NVARCH) -std=c++11 -dlink -o $@ $^  -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

$(CUSTOMNMS_ALL_GPU_LIB) : $(CUSTOMNMS_ALL_GPU_OBJ)
	$(NVCC) -arch=$(NVARCH) -std=c++11 -lib -o $@ $^  

%.o: %.cc
	$(CXX) $(CFLAGS) -c -o $@ $^ -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include

$(CUSTOMNMS_TARGET_LIB): $(CUSTOMNMS_GPU_OBJS) $(CUSTOMNMS_CPU_OBJS) 
	echo  $(CUSTOMNMS_GPU_OBJS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda-10.0/targets/x86_64-linux/lib -lcudart -lcudadevrt

customnms_test: tensorflow_customnms/python/ops/customnms_ops_test.py tensorflow_customnms/python/ops/customnms_ops.py $(CUSTOMNMS_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_customnms/python/ops/customnms_ops_test.py

customnms_sa: 
	nvcc -std=c++11 -g -O2  -o customnms_sa customnms_sa.cu

.PHONY : clean

clean:
	rm -f  *.o *.so *.a
