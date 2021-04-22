#ifndef __CONV_CUH__
#define __CONV_CUH__

//#include "../../include/curafft.h"
#include "../utils.h"
#include "../../include/dataType.h"

__global__ void conv_2d_nputsdriven(PCS *x, PCS *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, PCS es_c, PCS es_beta, int pirange, INT_M* cell_loc);

__global__ void conv_3d_nputsdriven(PCS *x, PCS *y, PCS *z, CUCPX *c, CUCPX *fw, int M,
    const int ns, int nf1, int nf2, int nf3, PCS es_c, PCS es_beta, int pirange, INT_M* cell_loc)

#endif