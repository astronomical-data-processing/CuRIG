#ifndef __UTILS_H__
#define __UTILS_H__

#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "dataType.h"

///contrib

// 0 correct  1 warning 2 error

#define CHECK(call)                                               \
  {                                                               \
    const cudaError_t error = call;                               \
    if (error != cudaSuccess)                                     \
    {                                                             \
      printf("Error:%s:%d", __FILE__, __LINE__);                  \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString); \
      exit(1);                                                    \
    }                                                             \
  }

#define M_1_2PI 0.159154943091895336 // 1/2/pi for faster calculation
#define M_2PI 6.28318530717958648    // 2 pi

#define PI (PCS) M_PI

#define EPSILON (double)1.1e-16

#define BLOCKSIZE 1024

#define SPEEDOFLIGHT 299792458.0

//random11 and rand01
// Random numbers: crappy unif random number generator in [0,1):
//#define rand01() (((PCS)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((PCS)rand() / RAND_MAX)
// unif[-1,1):
#define randm11() (2 * rand01() - (PCS)1.0)

int next235beven(int n, int b)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth") and is a multiple of b (b is a number that the only prime
// factors are 2,3,5). Adapted from fortran in hellskitchen. Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
// added condition about b Melody 05/31/20
{
  if (n <= 2)
    return 2;
  if (n % 2 == 1)
    n += 1;          // even
  int nplus = n - 2; // to cancel out the +=2 at start of loop
  int numdiv = 2;    // a dummy that is >1
  while ((numdiv > 1) || (nplus % b != 0))
  {
    nplus += 2; // stays even
    numdiv = nplus;
    while (numdiv % 2 == 0)
      numdiv /= 2; // remove all factors of 2,3,5...
    while (numdiv % 3 == 0)
      numdiv /= 3;
    while (numdiv % 5 == 0)
      numdiv /= 5;
  }
  return nplus;
}

void prefix_scan(PCS *d_arr, PCS *d_res, int n, int flag)
{
  /*
        n - number of elements
        flag - 1 inclusive, 0 exclusive
        Will the output at d_res
        thrust::inclusive_scan(d_arr, d_arr + n, d_res);???
    */
  thrust::device_ptr<PCS> d_ptr(d_arr); // not convert
  thrust::thrust::device_ptr<PCS> d_result(d_res);

  if (flag)
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result); // error may
  else
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
}

void get_max_min(PCS &max, PCS &min, PCS *d_array, int n)
{
  /*
        Get the maximum and minimum values of array by thrust
        Will be fast with one invokation getting max and min?
    */
  thrust::device_ptr<PCS> d_ptr = thrust::device_pointer_cast(d_array);
  max = *(thrust::max_element(d_ptr, d_ptr + n)); //revise
 
  min = *(thrust::min_element(d_ptr, d_ptr + n)); //revise
}

void GPU_info()
{
  /*
    int *h_max_test, *h_max_test2, *h_max_test3;
    CHECK(cudaMalloc(&h_max_test,sizeof(float)*1024*1024));
    CHECK(
    cudaMalloc(&h_max_test2,sizeof(float)*1024*1024));
    CHECK(
    cudaMalloc(&h_max_test3,sizeof(float)*1024*1024*1000));
    cudaFree(h_max_test);
    */
  printf("Starting... \n");
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess)
  {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    exit(EXIT_FAILURE);
  }

  if (deviceCount == 0)
    printf("There is no available device that support CUDA\n");
  else
    printf("Detected %d CUDA capable device(s)\n", deviceCount);

  int dev, driverVersion = 0, runtimeVersion = 0;

  dev = 0;

  printf("Input the device index:");
  scanf("%d", &dev);
  cudaSetDevice(dev);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Device %d: %s\n", dev, deviceProp.name);

  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("CUDA Driver Version / Runtime Version       %d.%d / %d.%d\n",
         driverVersion / 1000, driverVersion % 1000, runtimeVersion / 1000, runtimeVersion % 1000);
  printf("CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
  printf("Total amount of global memory:              %.2f MBytes\n", (float)deviceProp.totalGlobalMem / (pow(1024.0, 2)));
  printf("GPU clock rate:                             %.0f MHz\n", deviceProp.clockRate * 1e-3f);
  printf("Memory clock rate:                          %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
  printf("Memory Bus Width:                           %d-bit\n", deviceProp.memoryBusWidth);

  if (deviceProp.l2CacheSize)
  {
    printf("L2 Cache Size:                          %d bytes\n", deviceProp.l2CacheSize);
  }
  printf("Total amount of constant memory:            %lu bytes\n", deviceProp.totalConstMem);
  printf("Total amount of shared memory per block:    %lu bytes\n", deviceProp.sharedMemPerBlock);
  printf("Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
  printf("Warp size:                                  %d\n", deviceProp.warpSize);
  printf("Number of multiprocessors:                  %d\n", deviceProp.multiProcessorCount);
  printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
  //printf("Maximum number of blocks per multiprocessor: %d\n",deviceProp.maxBlocksPerMultiProcessor);
  printf("Maximum number of thread per block:          %d\n", deviceProp.maxThreadsPerBlock);
  printf("Maximum sizes of each dimension of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  printf("Maximum sizes of each dimension of a grid:  %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
         deviceProp.maxGridSize[2]);
  printf("Maximum memory pitch:                       %lu bytes\n", deviceProp.memPitch);
}

#endif