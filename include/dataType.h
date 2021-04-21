#ifndef __DATATYPE_H__
#define __DATATYPE_H__

/* ------------ data type definitions ----------------- */
#include <cuComplex.h>
#include <cuda.h>
#include <math.h>
#include <thrust/complex.h>
using namespace std::complex_literals;
#define COMPLEX(X) thrust::complex<X>

#undef PCS
#undef CPX

//define precision
#ifdef SINGLE
  #define PCS float
  #define CUCPX cuFloatComplex
  #define CUFFT_TYPE CUFFT_C2C
  #define CUFFT_EX cufftExecC2C
#else
  #define PCS double
  #define CUCPX cuDoubleComplex
  #define CUFFT_TYPE CUFFT_Z2Z
  #define CUFFT_EX cufftExecZ2Z
#endif

#define CPX COMPLEX(PCS)


#define INT_M int3


#endif

/*

#if (!defined(DATATYPES_H) && !defined(SINGLE)) || (!defined(DATATYPESF_H) && defined(SINGLE))
// Make sure we only include once per precision (as in finufft_eitherprec.h).
#ifndef SINGLE
#define DATATYPES_H
#else
#define DATATYPESF_H
#endif

// octave (mkoctfile) needs this otherwise it doesn't know what int64_t is!
#include <stdint.h>


// decide which kind of complex numbers to use in interface...
#ifdef __cplusplus
#include <complex>          // C++ type
#define COMPLEXIFY(X) std::complex<X>
#else
#include <complex.h>        // C99 type
#define COMPLEXIFY(X) X complex
#endif

#undef FLT
#undef CPX

// Precision-independent real and complex types for interfacing...
// (note these cannot be typedefs since we want dual-precision library)
#ifdef SINGLE
  #define FLT float
#else
  #define FLT double
#endif

#define CPX COMPLEXIFY(FLT)
*/