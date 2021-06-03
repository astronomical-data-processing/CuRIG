#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include "dataType.h"

void prefix_scan(PCS *d_arr, PCS *d_res, int n, int flag){
    /*
        n - number of elements
        flag - 1 inclusive, 0 exclusive
        Will the output at d_res
        thrust::inclusive_scan(d_arr, d_arr + n, d_res);???
    */
    thrust::device_ptr<PCS> d_ptr(d_arr); // not convert
	thrust::device_ptr<PCS> d_result(d_res);
    if(flag)
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
    else
	thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
}

void get_max_min(PCS &max, PCS &min, PCS *d_array, int n){
    /*
        Get the maximum and minimum values of array by thrust
        Will be fast with one invokation getting max and min?
    */
    thrust::device_ptr<PCS> d_ptr = thrust::device_pointer_cast(d_array);
    PCS *temp = thrust::max_element(d_ptr, d_ptr+n);
    max = *temp;
    temp = thrust::min_element(d_ptr, d_ptr+n);
    min = *temp;
}

void GPU_info(){
    /*
    int *h_max_test, *h_max_test2, *h_max_test3;
    CHECK(cudaMalloc(&h_max_test,sizeof(float)*1024*1024));
    CHECK(
    cudaMalloc(&h_max_test2,sizeof(float)*1024*1024));
    CHECK(
    cudaMalloc(&h_max_test3,sizeof(float)*1024*1024*1000));
    cudaFree(h_max_test);
    */
    printf("%s Starting... \n");
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess){
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    if (deviceCount==0) printf("There is no available device that support CUDA\n");
    else printf("Detected %d CUDA capable device(s)\n",deviceCount);

    int dev, driverVersion = 0, runtimeVersion = 0;

    dev = 0;

    printf("Input the device index:");
    scanf("%d",&dev);
    cudaSetDevice(dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Device %d: %s\n",dev,deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version / Runtime Version       %d.%d / %d.%d\n",
    driverVersion/1000,driverVersion%1000, runtimeVersion/1000, runtimeVersion%1000);
    printf("CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Total amount of global memory:              %.2f MBytes\n",(float)deviceProp.totalGlobalMem/(pow(1024.0,2)));
    printf("GPU clock rate:                             %.0f MHz\n",deviceProp.clockRate * 1e-3f);
    printf("Memory clock rate:                          %.0f MHz\n",deviceProp.memoryClockRate * 1e-3f);
    printf("Memory Bus Width:                           %d-bit\n", deviceProp.memoryBusWidth);

    if(deviceProp.l2CacheSize){
        printf("L2 Cache Size:                          %d bytes\n",deviceProp.l2CacheSize);
    }
    printf("Total amount of constant memory:            %lu bytes\n", deviceProp.totalConstMem);
    printf("Total amount of shared memory per block:    %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Total number of registers available per block: %d\n",deviceProp.regsPerBlock);
    printf("Warp size:                                  %d\n",deviceProp.warpSize);
    printf("Number of multiprocessors:                  %d\n", deviceProp.multiProcessorCount);
    printf("Maximum number of threads per multiprocessor: %d\n",deviceProp.maxThreadsPerMultiProcessor);
    //printf("Maximum number of blocks per multiprocessor: %d\n",deviceProp.maxBlocksPerMultiProcessor);
    printf("Maximum number of thread per block:          %d\n",deviceProp.maxThreadsPerBlock);
    printf("Maximum sizes of each dimension of a block: %d x %d x %d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1]
    ,deviceProp.maxThreadsDim[2]);
    printf("Maximum sizes of each dimension of a grid:  %d x %d x %d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],
    deviceProp.maxGridSize[2]);
    printf("Maximum memory pitch:                       %lu bytes\n",deviceProp.memPitch);

}