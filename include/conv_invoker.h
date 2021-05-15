#ifndef __CONV_INVOKER_H__
#define __CONV_INVOKER_H__


#include "curafft_plan.h"
#include "conv.h"

#define CONV_THREAD_NUM 32

int setup_conv_opts(conv_opts &c_opts, PCS eps, PCS upsampfac, int kerevalmeth);//cautious the &
int setup_plan(int nf1, int nf2, int M, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, curafft_plan *plan);
int curafft_conv(curafft_plan *plan);

#endif
