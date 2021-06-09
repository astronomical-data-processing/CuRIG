#ifndef __DFT_H__
#define __DFT_H__
		     

#include "curafft_plan.h"
int setup_plan(int nf1, int nf2, int nf3, int M, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, curafft_plan *plan);
void curadft_invoker(curafft_plan *plan);
#endif