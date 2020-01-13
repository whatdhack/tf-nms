/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "custom_nms.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <iostream>
#include <math.h>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.


template <typename T>
__global__ void CustomNms(const T *a, const T *b, T *c, int N) {

    // Static shmem calculation for convenience (Int 32x32 matrix)
    const int SHMEM_SIZE = 32 * 32;
    // Two statically-sized pieces of shared memory
    __shared__ T A[SHMEM_SIZE];
    __shared__ T B[SHMEM_SIZE];

    // Shorten these parameters for clean re-use
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Square tiles, so this could be blockDim.y as well
    int tile_size = blockDim.x;

    // Calculate global row and column positions for this thread
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    // Intermediate sum for element being written
    T tmp = 0;

    // Sweep tiles over entire matrix
    for (int i = 0; i < (N / tile_size); i++) {
        /*
            Every thread in a threadblock loads one element into shared memory
            The element location in shared memory corresponds to the thread's
            position in the threadblock (e.g. thread [0, 0] loads for 
            A[0 * tile_size + 0], and B[0 * tile_size + 0].)

            Explanation of indexing parameters
            For A:
                        row*N: Indexes the global row for this thread (loop-invariant)
                  i*tile_size: Indexes the new set of columns each iteration
                           tx: Indexes the column within that set
            for B:
                i*tile_size*N: Indexes the next set of rows each iteration
                         ty*N: Indexes the row within that set
                              col: Indexes the global column (loop-invariant)
        */
        A[ty * tile_size + tx] = a[(row * N) + (i * tile_size + tx)];
        B[ty * tile_size + tx] = b[(i * tile_size * N + ty * N) + col];

        // Ensure all threads have loaded their data before proceeding
        __syncthreads();

        // Calculate all temp values for this tile
        for (int j = 0; j < tile_size; j++) {
            tmp += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
        }

        // Ensure some threads don't progress and stomp current shared memory values
        __syncthreads();
    }

  // Write back the result
    c[row * N + col] = tmp;
}
// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct CustomNmsFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int size, const T* d_a, const T* d_b, T* d_c) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
	std::cout << "CustomNmsFunctor<GPUDevice, T>: size " << size << std::endl; 
    // Set the CTA and Grid dimensions
    int THREADS = 32;
	int N = sqrt(size);
	if ( N < THREADS ) THREADS = N;
    int BLOCKS = size / (THREADS*THREADS);
	std::cout << "CustomNmsFunctor<GPUDevice, T>: N " << N << " THREADS "<< THREADS 
              << " BLOCKS " << BLOCKS << std::endl; 

    // Use dim3 objects for 2-D grids and threadblocks
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);
    CustomNms<T><<<blocks, threads, 0, d.stream()>>>(d_a, d_b, d_c, N);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct CustomNmsFunctor<GPUDevice, int32>;
template struct CustomNmsFunctor<GPUDevice, float>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
