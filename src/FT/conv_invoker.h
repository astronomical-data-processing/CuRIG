#ifndef __CONV_INVOKER_H__
#define __CONV_INVOKER_H__

#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>
#include "../../include/dataType.h"
#include "../utils.h"

#define MAX_KERNEL_WIDTH 16     // w, even when padded
                           // (see evaluate_kernel_vector); also for common
#define CONV_THREAD_NUM 32

// NU coord handling macro: if p is true, rescales from [-pi,pi] to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
//#define RESCALE(x,N,p) (p ? \
		     ((x*M_1_2PI + (x<-PI ? 1.5 : (x>=PI ? -0.5 : 0.5)))*N) : \
		     (x<0 ? x+N : (x>=N ? x-N : x)))
// yuk! But this is *so* much faster than slow std::fmod that we stick to it.






int setup_conv_opts(conv_opts &opts, PCS eps, PCS upsampfac, int kerevalmeth);
int setup_plan(int nf1, int nf2, int M, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, curafft_plan *plan);
int curafft_conv(curafft_plan *plan);

#endif
