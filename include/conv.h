#ifndef __CONV_CUH__
#define __CONV_CUH__

#include <stdlib.h>
#include "utils.h"
#include "dataType.h"
#include "curafft_plan.h"

// NU coord handling macro: if p is true, rescales from [-pi,pi) to [0,N], then
// folds *only* one period below and above, ie [-N,2N], into the domain [0,N]...
// if p is false, rescale from [min, max) to [0,N) 
// Part of RESCALE is from Barnett 2/7/17.
#define RESCALE(x,N,p,max,min) (p ? \
		     ((x*M_1_2PI + (x<-PI ? 1.5 : (x>=PI ? -0.5 : 0.5)))*N) : \
             (x>=max ? (x+min-max) : ((x - min) / (max - min) * N))) 

__global__ void conv_1d_nputsdriven(PCS *x, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, PCS es_c, PCS es_beta, int pirange, PCS *max, PCS *min, INT_M* cell_loc);

__global__ void conv_2d_nputsdriven(PCS *x, PCS *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, PCS es_c, PCS es_beta, int pirange, PCS *max, PCS *min, INT_M* cell_loc);


__global__ void conv_3d_nputsdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int M,
    const int ns, int nf1, int nf2, int nf3, PCS es_c, PCS es_beta, int pirange, PCS *max, PCS *min, INT_M* cell_loc);

#endif