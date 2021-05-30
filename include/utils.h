#ifndef __UTILS_H__
#define __UTILS_H__

#include <cstdlib>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "dataType.h"

///contrib


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

#define SPEEDOFLIGHT 299792458.0

//random11 and rand01
// Random numbers: crappy unif random number generator in [0,1):
//#define rand01() (((PCS)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((PCS)rand()/RAND_MAX)
// unif[-1,1):
#define randm11() (2*rand01() - (PCS)1.0)

int next235beven(int n, int b)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth") and is a multiple of b (b is a number that the only prime 
// factors are 2,3,5). Adapted from fortran in hellskitchen. Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
// added condition about b Melody 05/31/20
{
  if (n<=2) return 2;
  if (n%2 == 1) n+=1;   // even
  int nplus = n-2;   // to cancel out the +=2 at start of loop
  int numdiv = 2;    // a dummy that is >1
  while ((numdiv>1) || (nplus%b != 0)) {
    nplus += 2;         // stays even
    numdiv = nplus;
    while (numdiv%2 == 0) numdiv /= 2;  // remove all factors of 2,3,5...
    while (numdiv%3 == 0) numdiv /= 3;
    while (numdiv%5 == 0) numdiv /= 5;
  }
  return nplus;
}

#endif