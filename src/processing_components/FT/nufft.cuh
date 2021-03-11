#ifndef NUFFT_CUH
#define NUFFT_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cufft.h>
#include "matrix.cuh"
#include <thrust/complex.h>
using namespace thrust;
using namespace std::complex_literals;
//using namespace std::cout;
#define THREADNUM 32

void directFT_1d(int M, complex<float> *c, float *x, int length, int direction, float df);


#endif