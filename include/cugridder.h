#ifndef __CUGRIDDER_H__
#define __CUGRIDDER_H__

#include "curafft_plan.h"
#include "ragridder_plan.h"
#include "utils.h"

int gridder_setting(int N1, int N2, int method, int kerevalmeth, int w_term_method, int direction, double sigma, int iflag,
    int batchsize, int M, PCS fov, int channel, visibility *pointer_v, PCS *d_u, PCS *d_v, PCS *d_w, CUCPX *d_c, curafft_plan *plan, ragridder_plan *gridder_plan);
int gridder_exectuion(curafft_plan *plan, ragridder_plan *gridder_pan);
int gridder_destroy(curafft_plan *plan, ragridder_plan *gridder_plan);
#endif