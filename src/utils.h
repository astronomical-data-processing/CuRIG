#ifndef __UTILS_H__
#define __UTILS_H__

#include <cstdlib>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "../include/dataType.h"
#include "../include/curafft_plan.h"
#include "../include/curafft_opts.h"
#include "../src/FT/conv_invoker.h"
#include "../src/RA/visibility.h"



// 0 correct  1 warning 2 error

#define CHECK(call)                                                     \
{                                                                   \
        const cudaError_t error = call;                                 \
        if (error != cudaSuccess)                                       \
        {                                                               \
            printf("Error:%s:%d", __FILE__, __LINE__);                  \
            printf("code:%d, reason: %s\n", error, cudaGetErrorString); \
            exit(1);                                                    \
        }                                                               \
}

#define M_1_2PI 0.159154943091895336 // 1/2/pi for faster calculation
#define M_2PI   6.28318530717958648  // 2 pi

#define PI (PCS)M_PI

#define EPSILON (double)1.1e-16

#define BLOCKSIZE 1024

//random11 and rand01
// Random numbers: crappy unif random number generator in [0,1):
//#define rand01() (((FLT)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((PCS)rand()/RAND_MAX)
// unif[-1,1]:
#define randm11() (2*rand01() - (PCS)1.0)

#endif