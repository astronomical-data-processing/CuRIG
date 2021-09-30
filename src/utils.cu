/*
  Utility functions
  1. prefix_scan
  2. get_max_min
  3. rescale
  4. shift_and_scale
  5. matrix transpose
*/

#include "utils.h"
#include "common_utils.h"
#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <iostream>
#include <stdio.h>
//#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "datatype.h"

void prefix_scan(PCS *d_arr, PCS *d_res, int n, int flag)
{
  /*
    n - number of elements
    flag - 1 inclusive, 0 exclusive
    thrust::inclusive_scan(d_arr, d_arr + n, d_res);
  */
  thrust::device_ptr<PCS> d_ptr(d_arr); // not convert
  thrust::device_ptr<PCS> d_result(d_res);

  if (flag)
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
  else
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
}

void get_max_min(PCS &max, PCS &min, PCS *d_array, int n)
{
  /*
    Get the maximum and minimum of array by thrust
    d_array - array on device
    n - length of array
  */
  thrust::device_ptr<PCS> d_ptr = thrust::device_pointer_cast(d_array);
  max = *(thrust::max_element(d_ptr, d_ptr + n));

  min = *(thrust::min_element(d_ptr, d_ptr + n));
}

// real and complex array scaling
__global__ void rescaling_real(PCS *x, PCS scale_ratio, int N)
{
  int idx;
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    x[idx] *= scale_ratio;
  }
}

__global__ void rescaling_complex(CUCPX *x, PCS scale_ratio, int N)
{
  int idx;
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    x[idx].x *= scale_ratio;
    x[idx].y *= scale_ratio;
  }
}

void rescaling_real_invoker(PCS *d_x, PCS scale_ratio, int N)
{
  int blocksize = 512;
  rescaling_real<<<(N - 1) / blocksize + 1, blocksize>>>(d_x, scale_ratio, N);
  CHECK(cudaDeviceSynchronize());
}

void rescaling_complex_invoker(CUCPX *d_x, PCS scale_ratio, int N)
{
  int blocksize = 512;
  rescaling_complex<<<(N - 1) / blocksize + 1, blocksize>>>(d_x, scale_ratio, N);
  CHECK(cudaDeviceSynchronize());
}

__global__ void shift_and_scale(PCS i_center, PCS o_center, PCS gamma, PCS *d_u, PCS *d_x, int M, int N)
{
  int idx;
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += gridDim.x * blockDim.x)
  {
    d_u[idx] = (d_u[idx] - i_center) / gamma;
  }
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    d_x[idx] = (d_x[idx] - o_center) * gamma;
  }
}

void shift_and_scale_invoker(PCS i_center, PCS o_center, PCS gamma, PCS *d_u, PCS *d_x, int M, int N)
{
  // Specified for nu to nu fourier transform
  int blocksize = 512;
  shift_and_scale<<<(max(M, N) - 1) / blocksize + 1, blocksize>>>(i_center, o_center, gamma, d_u, d_x, M, N);
  CHECK(cudaDeviceSynchronize());
}

__global__ void transpose(PCS *odata, PCS *idata, int width, int height)
{
  //* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
  // refer https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
  __shared__ PCS block[BLOCKSIZE][BLOCKSIZE];

  // read the matrix tile into shared memory
  // load one element per thread from device memory (idata) and store it
  // in transposed order in block[][]
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //height
  unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //width
  if ((yIndex < width) && (xIndex < height))
  {
    unsigned int index_in = xIndex * width + yIndex;
    block[threadIdx.x][threadIdx.y] = idata[index_in];
  }

  // synchronise to ensure all writes to block[][] have completed
  __syncthreads();

  // write the transposed matrix tile to global memory (odata) in linear order
  xIndex = blockIdx.y * blockDim.x + threadIdx.x;
  yIndex = blockIdx.x * blockDim.y + threadIdx.y;
  if ((yIndex < height) && (xIndex < width))
  {
    unsigned int index_out = xIndex * height + yIndex;
    odata[index_out] = block[threadIdx.y][threadIdx.x];
  }
  // __syncthreads();
}

int matrix_transpose_invoker(PCS *d_arr, int width, int height)
{
  int ier = 0;
  int blocksize = BLOCKSIZE;
  dim3 block(blocksize, blocksize);
  dim3 grid((height - 1) / blocksize + 1, (width - 1) / blocksize + 1);
  PCS *temp_o;
  checkCudaErrors(cudaMalloc((void **)&temp_o, sizeof(PCS) * width * height));
  transpose<<<grid, block>>>(temp_o, d_arr, width, height);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(d_arr, temp_o, sizeof(PCS) * width * height, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(temp_o));
  return ier;
}

__global__ void matrix_elementwise_multiply(CUCPX *a, PCS *b, int N)
{
  int idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    a[idx].x = a[idx].x * b[idx];
    a[idx].y = a[idx].y * b[idx];
  }
}

int matrix_elementwise_multiply_invoker(CUCPX *a, PCS *b, int N)
{
  int ier = 0;
  int blocksize = 512;
  matrix_elementwise_multiply<<<(N - 1) / blocksize + 1, blocksize>>>(a, b, N);
  checkCudaErrors(cudaDeviceSynchronize());
  return ier;
}

__global__ void matrix_elementwise_divide(CUCPX *a, PCS *b, int N)
{
  int idx;
  for (idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    a[idx].x = a[idx].x / b[idx];
    a[idx].y = a[idx].y / b[idx];
  }
}

int matrix_elementwise_divide_invoker(CUCPX *a, PCS *b, int N)
{
  int ier = 0;
  int blocksize = 512;
  matrix_elementwise_multiply<<<(N - 1) / blocksize + 1, blocksize>>>(a, b, N);
  checkCudaErrors(cudaDeviceSynchronize());
  return ier;
}